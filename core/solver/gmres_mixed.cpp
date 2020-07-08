/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/gmres_mixed.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include <iostream>


#include "core/solver/gmres_mixed_accessor.hpp"
#include "core/solver/gmres_mixed_kernels.hpp"


//#define TIMING 1


#ifdef TIMING
using double_seconds = std::chrono::duration<double>;
#define TIMING_STEPS 1
#endif


namespace gko {
namespace solver {


namespace gmres_mixed {


GKO_REGISTER_OPERATION(initialize_1, gmres_mixed::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, gmres_mixed::initialize_2);
GKO_REGISTER_OPERATION(step_1, gmres_mixed::step_1);
GKO_REGISTER_OPERATION(step_2, gmres_mixed::step_2);


}  // namespace gmres_mixed


template <typename ValueType, typename ValueTypeKrylovBases>
void GmresMixed<ValueType, ValueTypeKrylovBases>::apply_impl(const LinOp *b,
                                                             LinOp *x) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    using Vector = matrix::Dense<ValueType>;
    using VectorNorms = matrix::Dense<remove_complex<ValueType>>;
    using LowArray = Array<ValueTypeKrylovBases>;
    using KrylovAccessor = kernels::Accessor3d<ValueTypeKrylovBases, ValueType>;
    using Accessor3dHelper =
        kernels::Accessor3dHelper<ValueType, ValueTypeKrylovBases>;


    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto residual = Vector::create_with_config_of(dense_b);
    const dim<3> krylov_bases_dim{
        krylov_dim_ + 1, system_matrix_->get_size()[1], dense_b->get_size()[1]};
    // const dim<2> krylov_bases_dim{
    //     system_matrix_->get_size()[1],
    //     (krylov_dim_ + 1) * dense_b->get_size()[1]};
    // const size_type krylov_bases_stride = krylov_bases_dim[1];
    // LowArray krylov_bases(exec, krylov_bases_dim[0] * krylov_bases_stride);
    // KrylovAccessor krylov_bases_accessor(krylov_bases.get_data(),
    //                                     krylov_bases_stride);
    Accessor3dHelper helper(exec, krylov_bases_dim);
    auto krylov_bases_accessor = helper.get_accessor();

    auto next_krylov_basis = Vector::create_with_config_of(dense_b);
    std::shared_ptr<matrix::Dense<ValueType>> preconditioned_vector =
        Vector::create_with_config_of(dense_b);
    auto hessenberg = Vector::create(
        exec, dim<2>{krylov_dim_ + 1, krylov_dim_ * dense_b->get_size()[1]});
    auto buffer =
        Vector::create(exec, dim<2>{krylov_dim_ + 1, dense_b->get_size()[1]});
    auto givens_sin =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto givens_cos =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto residual_norm_collection =
        Vector::create(exec, dim<2>{krylov_dim_ + 1, dense_b->get_size()[1]});
    auto residual_norm =
        VectorNorms::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto b_norm = VectorNorms::create(exec, dim<2>{1, dense_b->get_size()[1]});
    // TODO: write description what the different rows represent
    // The optional entry stores the infinity_norm of each next_krylov_vector,
    // which is only used to compute the scale
    auto arnoldi_norm = VectorNorms::create(
        exec, dim<2>{2 + Accessor3dHelper::Accessor::has_scale,
                     dense_b->get_size()[1]});
    Array<size_type> final_iter_nums(this->get_executor(),
                                     dense_b->get_size()[1]);
    auto y = Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});

    bool one_changed{};
    Array<stopping_status> stop_status(this->get_executor(),
                                       dense_b->get_size()[1]);
    Array<stopping_status> reorth_status(this->get_executor(),
                                         dense_b->get_size()[1]);
    Array<size_type> num_reorth(this->get_executor(), dense_b->get_size()[1]);
    int num_restarts = 0, num_reorth_steps = 0, num_reorth_vectors = 0;

    // std::cout << "Before initializate_1" << std::endl;
    // Initialization
    exec->run(gmres_mixed::make_initialize_1(
        dense_b, b_norm.get(), residual.get(), givens_sin.get(),
        givens_cos.get(), &stop_status, krylov_dim_));
    // b_norm = norm(b)
    // residual = dense_b
    // givens_sin = givens_cos = 0
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          residual.get());
    // residual = residual - Ax

    // std::cout << "Before initializate_2" << std::endl;
    exec->run(gmres_mixed::make_initialize_2(
        residual.get(), residual_norm.get(), residual_norm_collection.get(),
        arnoldi_norm.get(), krylov_bases_accessor, next_krylov_basis.get(),
        &final_iter_nums, krylov_dim_));
    // residual_norm = norm(residual)
    // residual_norm_collection = {residual_norm, 0, ..., 0}
    // krylov_bases(:, 1) = residual / residual_norm
    // next_krylov_basis = residual / residual_norm
    // final_iter_nums = {0, ..., 0}

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, residual.get());

    int total_iter = -1;
    size_type restart_iter = 0;

    auto before_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(dense_x);
    auto after_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(dense_x);

#ifdef TIMING
    exec->synchronize();
    auto start = std::chrono::steady_clock::now();
#ifdef TIMING_STEPS
    auto time_RSTRT = start - start;
    auto time_SPMV = start - start;
    auto time_STEP1 = start - start;
#endif
#endif
    bool stop_already_encountered{false};
    // std::cout << "Before loop" << std::endl;
    while (true) {
        ++total_iter;
        this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual.get(), dense_x, residual_norm.get());
        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual.get())
                .residual_norm(residual_norm.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            if (stop_already_encountered) {
                break;
            }
            stop_already_encountered = true;
            Array<stopping_status> host_stop_status(
                this->get_executor()->get_master(), stop_status);
            bool host_array_changed{false};
            for (size_type i = 0; i < host_stop_status.get_num_elems(); ++i) {
                auto local_status = host_stop_status.get_data() + i;
                if (local_status->has_converged()) {
                    local_status->reset();
                    host_array_changed = true;
                }
            }
            if (host_array_changed) {
                stop_status = host_stop_status;
            } else {
                break;
            }
        } else {
            stop_already_encountered = false;
        }

        if (stop_already_encountered || restart_iter == krylov_dim_) {
#ifdef TIMING_STEPS
            exec->synchronize();
            auto t_aux_0 = std::chrono::steady_clock::now();
#endif
            //            std::cout << "RESTARTING" << std::endl;
            num_restarts++;
            // Restart
            /*
            std::cout << "[ ";
            for (int i = 0; i <= krylov_dim_; i++) {
                for (int j = 0; j < krylov_dim_ * dense_b->get_size()[1];
                     j++) {
                    if (j > 0) std::cout << ", ";
                    std::cout << hessenberg->at(i, j);
                }
                std::cout << std::endl;
            }
            std::cout << "]" << std::endl;
            */
            // use a view in case this is called earlier
            auto hessenberg_view = hessenberg->create_submatrix(
                span{0, restart_iter},
                span{0, dense_b->get_size()[1] * (restart_iter)});

            exec->run(gmres_mixed::make_step_2(
                residual_norm_collection.get(),
                krylov_bases_accessor.to_const(), hessenberg_view.get(),
                y.get(), before_preconditioner.get(), &final_iter_nums));
            /* */ /*
            auto hessenberg_small = hessenberg->create_submatrix(
                span{0, restart_iter},
                span{0, dense_b->get_size()[1] * (restart_iter)});
            exec->run(gmres_mixed::make_step_2(
                residual_norm_collection.get(),
                krylov_bases_accessor.to_const(),
                hessenberg_small.get(), y.get(),
                before_preconditioner.get(), &final_iter_nums));
            */
            // Solve upper triangular.
            // y = hessenberg \ residual_norm_collection

            get_preconditioner()->apply(before_preconditioner.get(),
                                        after_preconditioner.get());
            dense_x->add_scaled(one_op.get(), after_preconditioner.get());
            // Solve x
            // x = x + get_preconditioner() * krylov_bases * y
            residual->copy_from(dense_b);
            // residual = dense_b
            system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                                  residual.get());
            // residual = residual - Ax
            exec->run(gmres_mixed::make_initialize_2(
                residual.get(), residual_norm.get(),
                residual_norm_collection.get(), arnoldi_norm.get(),
                krylov_bases_accessor, next_krylov_basis.get(),
                &final_iter_nums, krylov_dim_));
            // residual_norm = norm(residual)
            // residual_norm_collection = {residual_norm, 0, ..., 0}
            // krylov_bases(:, 1) = residual / residual_norm
            // next_krylov_basis = residual / residual_norm
            // final_iter_nums = {0, ..., 0}
            restart_iter = 0;
#ifdef TIMING_STEPS
            exec->synchronize();
            time_RSTRT += std::chrono::steady_clock::now() - t_aux_0;
#endif
        }

        get_preconditioner()->apply(next_krylov_basis.get(),
                                    preconditioned_vector.get());
        // preconditioned_vector = get_preconditioner() * next_krylov_basis

        // Do Arnoldi and givens rotation
        auto hessenberg_iter = hessenberg->create_submatrix(
            span{0, restart_iter + 2},
            span{dense_b->get_size()[1] * restart_iter,
                 dense_b->get_size()[1] * (restart_iter + 1)});
        auto buffer_iter = buffer->create_submatrix(
            span{0, restart_iter + 2}, span{0, dense_b->get_size()[1]});

#ifdef TIMING_STEPS
        exec->synchronize();
        auto t_aux_1 = std::chrono::steady_clock::now();
#endif
        // Start of arnoldi
        system_matrix_->apply(preconditioned_vector.get(),
                              next_krylov_basis.get());
        // next_krylov_basis = A * preconditioned_vector
#ifdef TIMING_STEPS
        exec->synchronize();
        time_SPMV += std::chrono::steady_clock::now() - t_aux_1;
#endif

#ifdef TIMING_STEPS
        exec->synchronize();
        auto t_aux_2 = std::chrono::steady_clock::now();
#endif
        exec->run(gmres_mixed::make_step_1(
            next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
            residual_norm.get(), residual_norm_collection.get(),
            krylov_bases_accessor, hessenberg_iter.get(), buffer_iter.get(),
            b_norm.get(), arnoldi_norm.get(), restart_iter, &final_iter_nums,
            &stop_status, &reorth_status, &num_reorth, &num_reorth_steps,
            &num_reorth_vectors));
#ifdef TIMING_STEPS
        exec->synchronize();
        time_STEP1 += std::chrono::steady_clock::now() - t_aux_2;
#endif
        // for i in 0:restart_iter
        //     hessenberg(restart_iter, i) = next_krylov_basis' *
        //     krylov_bases(:, i) next_krylov_basis  -= hessenberg(restart_iter,
        //     i) * krylov_bases(:, i)
        // end
        // hessenberg(restart_iter, restart_iter + 1) = norm(next_krylov_basis)
        // next_krylov_basis /= hessenberg(restart_iter, restart_iter + 1)
        // End of arnoldi
        // Start apply givens rotation
        // for j in 0:restart_iter
        //     temp             =  cos(j)*hessenberg(j) +
        //                         sin(j)*hessenberg(j+1)
        //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
        //                         cos(j)*hessenberg(j+1)
        //     hessenberg(j)    =  temp;
        // end
        // Calculate sin and cos
        // hessenberg(restart_iter)   =
        // cos(restart_iter)*hessenberg(restart_iter) +
        //                      sin(restart_iter)*hessenberg(restart_iter)
        // hessenberg(restart_iter+1) = 0
        // End apply givens rotation
        // Calculate residual norm

        restart_iter++;
    }

    // Solve x
#ifdef TIMING_STEPS
    exec->synchronize();
    auto t_aux_3 = std::chrono::steady_clock::now();
#endif
    /*
    auto krylov_bases_small = krylov_bases->create_submatrix(
        span{0, system_matrix_->get_size()[0]},
        span{0, dense_b->get_size()[1] * (restart_iter + 1)});
    const dim<2> krylov_bases_dim{
        system_matrix_->get_size()[1],
        (krylov_dim_ + 1) * dense_b->get_size()[1]};
    */
    /*
    const dim<2> krylov_bases_small_dim{
        system_matrix_->get_size()[0],
        dense_b->get_size()[1] * (restart_iter + 1)};
    KrylovAccessor krylov_bases_small_accessor{krylov_bases.get_data(),
                                               krylov_bases_stride};
    */

    auto hessenberg_small = hessenberg->create_submatrix(
        span{0, restart_iter},
        span{0, dense_b->get_size()[1] * (restart_iter)});

    exec->run(gmres_mixed::make_step_2(
        residual_norm_collection.get(), krylov_bases_accessor.to_const(),
        hessenberg_small.get(), y.get(), before_preconditioner.get(),
        &final_iter_nums));
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
#ifdef TIMING_STEPS
    exec->synchronize();
    auto time_STEP2 = std::chrono::steady_clock::now() - t_aux_3;
#endif

#ifdef TIMING_STEPS
    exec->synchronize();
    auto t_aux_4 = std::chrono::steady_clock::now();
#endif
    get_preconditioner()->apply(before_preconditioner.get(),
                                after_preconditioner.get());
    dense_x->add_scaled(one_op.get(), after_preconditioner.get());
#ifdef TIMING_STEPS
    exec->synchronize();
    auto time_SOLVEX = std::chrono::steady_clock::now() - t_aux_4;
#endif
    // Solve x
    // x = x + get_preconditioner() * krylov_bases * y

#ifdef TIMING
    exec->synchronize();
    auto time = std::chrono::steady_clock::now() - start;
#endif
#ifdef TIMING
    std::cout << "total_iter = " << total_iter << std::endl;
    std::cout << "num_restarts = " << num_restarts << std::endl;
    std::cout << "reorth_steps = " << num_reorth_steps << std::endl;
    std::cout << "reorth_vectors = " << num_reorth_vectors << std::endl;
    std::cout << "time = "
              << std::chrono::duration_cast<double_seconds>(time).count()
              << std::endl;
#ifdef TIMING_STEPS
    std::cout << "time_RSTRT = "
              << std::chrono::duration_cast<double_seconds>(time_RSTRT).count()
              << std::endl;
    std::cout << "time_SPMV = "
              << std::chrono::duration_cast<double_seconds>(time_SPMV).count()
              << std::endl;
    std::cout << "time_STEP1 = "
              << std::chrono::duration_cast<double_seconds>(time_STEP1).count()
              << std::endl;
    std::cout << "time_STEP2 = "
              << std::chrono::duration_cast<double_seconds>(time_STEP2).count()
              << std::endl;
    std::cout << "time_SOLVEX = "
              << std::chrono::duration_cast<double_seconds>(time_SOLVEX).count()
              << std::endl;
#endif
    write(std::cout, lend(residual_norm));
#endif
}


/*
template <typename ValueType, typename ValueTypeKrylovBases, bool
Reorthogonalization, bool MGS_CGS> void
GmresMixed<ValueType,ValueTypeKrylovBases,
                Reorthogonalization,MGS_CGS>::apply_impl(const LinOp *alpha,
const LinOp *b, const LinOp *residual_norm_collection, LinOp *x) const
*/
template <typename ValueType, typename ValueTypeKrylovBases>
void GmresMixed<ValueType, ValueTypeKrylovBases>::apply_impl(
    const LinOp *alpha, const LinOp *b, const LinOp *residual_norm_collection,
    LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(residual_norm_collection);
    dense_x->add_scaled(alpha, x_clone.get());
}

/*
#define GKO_DECLARE_GMRES_MIXED_BOOL(_type1, _type2, _bool1, _bool2)
      class GmresMixed<_type1, _type2, _bool1, _bool2>
GKO_INSTANTIATE_FOR_EACH_MIXED_BOOL_TYPE(GKO_DECLARE_GMRES_MIXED_BOOL);
*/
#define GKO_DECLARE_GMRES_MIXED(_type1, _type2) class GmresMixed<_type1, _type2>
GKO_INSTANTIATE_FOR_EACH_GMRES_MIXED_TYPE(GKO_DECLARE_GMRES_MIXED);


}  // namespace solver
}  // namespace gko
