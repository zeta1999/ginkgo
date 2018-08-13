/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/gmres.hpp"


#include "core/base/array.hpp"
#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/name_demangling.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"

#include <iostream>


namespace gko {
namespace solver {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize_1, gmres::initialize_1<ValueType>);
    GKO_REGISTER_OPERATION(initialize_2, gmres::initialize_2<ValueType>);
    GKO_REGISTER_OPERATION(step_1, gmres::step_1<ValueType>);
    GKO_REGISTER_OPERATION(step_2, gmres::step_2<ValueType>);
};


}  // namespace


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    this->template log<log::Logger::apply>(GKO_FUNCTION_NAME);

    using std::swap;
    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto residual = Vector::create_with_config_of(dense_b);
    auto Krylov_bases = Vector::create(
        exec,
        dim<2>{system_matrix_->get_size()[1],
               (solver::default_max_iter_num + 1) * dense_b->get_size()[1]});
    auto Hessenberg = Vector::create(
        exec, dim<2>{solver::default_max_iter_num + 1,
                     solver::default_max_iter_num * dense_b->get_size()[1]});
    auto givens_sin = Vector::create(
        exec, dim<2>{solver::default_max_iter_num, dense_b->get_size()[1]});
    auto givens_cos = Vector::create(
        exec, dim<2>{solver::default_max_iter_num, dense_b->get_size()[1]});
    auto residual_norms = Vector::create(
        exec, dim<2>{solver::default_max_iter_num + 1, dense_b->get_size()[1]});
    auto residual_norm =
        Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto b_norm = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    Array<size_type> final_iter_nums(this->get_executor(),
                                     dense_b->get_size()[1]);

    bool one_changed{};
    Array<stopping_status> stop_status(this->get_executor(),
                                       dense_b->get_size()[1]);

    // Initialization
    exec->run(TemplatedOperation<ValueType>::make_initialize_1_operation(
        dense_b, b_norm.get(), residual.get(), givens_sin.get(),
        givens_cos.get(), &final_iter_nums, &stop_status,
        solver::default_max_iter_num));
    // b_norm = norm(b)
    // residual = dense_b
    // givens_sin = givens_cos = 0
    // final_iter_nums = {0, ..., 0}
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          residual.get());
    // residual = residual(b) - Ax

    exec->run(TemplatedOperation<ValueType>::make_initialize_2_operation(
        residual.get(), residual_norm.get(), residual_norms.get(),
        Krylov_bases.get(), solver::default_max_iter_num));
    // residual_norm = norm(residual)
    // residual_norms = {residual_norm, 0, ..., 0}
    // Krylov_bases(:, 1) = residual / residual_norm

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, residual.get());

    size_type iter = 0;
    for (; iter < solver::default_max_iter_num; ++iter) {
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual_norm(residual_norm.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            this->template log<log::Logger::converged>(iter + 1,
                                                       residual.get());
            break;
        }

        for (int i = 0; i < dense_b->get_size()[1]; ++i) {
            final_iter_nums.get_data()[i] +=
                (1 - stop_status.get_const_data()[i].has_stopped());
        }

        // Do Arnoldi and givens rotation
        auto Krylov_bases_iter = Krylov_bases->create_submatrix(
            span{0, system_matrix_->get_size()[0]},
            span{dense_b->get_size()[1] * iter,
                 dense_b->get_size()[1] * (iter + 1)});
        auto Hessenberg_iter = Hessenberg->create_submatrix(
            span{0, iter + 2}, span{dense_b->get_size()[1] * iter,
                                    dense_b->get_size()[1] * (iter + 1)});
        auto next_Krylov_basis = Vector::create_with_config_of(dense_b);

        system_matrix_->apply(Krylov_bases_iter.get(), next_Krylov_basis.get());
        // next_Krylov_basis = A * Krylov_bases(:, iter)
        exec->run(TemplatedOperation<ValueType>::make_step_1_operation(
            next_Krylov_basis.get(), givens_sin.get(), givens_cos.get(),
            residual_norm.get(), residual_norms.get(), Krylov_bases.get(),
            Hessenberg_iter.get(), b_norm.get(), iter, &stop_status));

        this->template log<log::Logger::iteration_complete>(iter + 1);
    }

    // Solve x
    auto y = Vector::create(
        exec, dim<2>{solver::default_max_iter_num, dense_b->get_size()[1]});
    auto Krylov_bases_small = Krylov_bases->create_submatrix(
        span{0, system_matrix_->get_size()[0]},
        span{0, dense_b->get_size()[1] * (iter + 1)});
    auto Hessenberg_small = Hessenberg->create_submatrix(
        span{0, iter}, span{0, dense_b->get_size()[1] * (iter)});

    exec->run(TemplatedOperation<ValueType>::make_step_2_operation(
        residual_norms.get(), Krylov_bases_small.get(), Hessenberg_small.get(),
        y.get(), dense_x, &final_iter_nums));
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *residual_norms, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(residual_norms);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_GMRES(_type) class Gmres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);
#undef GKO_DECLARE_GMRES


}  // namespace solver
}  // namespace gko
