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

#include <ginkgo/core/factorization/lu.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/lu_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace lu_factorization {


GKO_REGISTER_OPERATION(add_diagonal_elements,
                       lu_factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       lu_factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, lu_factorization::initialize_l_u);
GKO_REGISTER_OPERATION(compute_l_u_factors,
                       lu_factorization::compute_l_u_factors);
GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);


}  // namespace lu_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Lu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp> &system_matrix, bool skip_sorting,
    std::shared_ptr<typename l_matrix_type::strategy_type> l_strategy,
    std::shared_ptr<typename u_matrix_type::strategy_type> u_strategy) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();

    // Only copies the matrix if it is not on the same executor or was not in
    // the right format. Throws an exception if it is not convertable.
    std::unique_ptr<CsrMatrix> csr_system_matrix_unique_ptr{};
    auto csr_system_matrix_const =
        dynamic_cast<const CsrMatrix *>(system_matrix.get());
    CsrMatrix *csr_system_matrix{};
    if (csr_system_matrix_const == nullptr ||
        csr_system_matrix_const->get_executor() != exec) {
        csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
        as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
            ->convert_to(csr_system_matrix_unique_ptr.get());
    } else {
        csr_system_matrix_unique_ptr = csr_system_matrix_const->clone();
    }
    csr_system_matrix = csr_system_matrix_unique_ptr.get();
    // If it needs to be sorted, copy it if necessary and sort it
    if (!skip_sorting) {
        if (csr_system_matrix_unique_ptr == nullptr) {
            csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
            csr_system_matrix_unique_ptr->copy_from(csr_system_matrix);
        }
        csr_system_matrix_unique_ptr->sort_by_column_index();
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }

    // Add explicit diagonal zero elements if they are missing
    exec->run(
        lu_factorization::make_add_diagonal_elements(csr_system_matrix, true));

    const auto matrix_size = csr_system_matrix->get_size();
    const auto number_rows = matrix_size[0];

    // TODO : Need to set the csr strategies later
    std::shared_ptr<CsrMatrix> l_factor =
        l_matrix_type::create(exec, matrix_size);
    std::shared_ptr<CsrMatrix> u_factor =
        u_matrix_type::create(exec, matrix_size);

    exec->run(lu_factorization::make_compute_l_u_factors(
        csr_system_matrix, l_factor.get(), u_factor.get()));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}


#define GKO_DECLARE_LU(ValueType, IndexType) class Lu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU);


}  // namespace factorization
}  // namespace gko
