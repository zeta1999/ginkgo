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

#include <ginkgo/core/matrix/diagonal.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/diagonal_kernels.hpp"


namespace gko {
namespace matrix {
namespace diagonal {


GKO_REGISTER_OPERATION(apply_to_dense, diagonal::apply_to_dense);
GKO_REGISTER_OPERATION(right_apply_to_dense, diagonal::right_apply_to_dense);
GKO_REGISTER_OPERATION(apply_to_csr, diagonal::apply_to_csr);
GKO_REGISTER_OPERATION(right_apply_to_csr, diagonal::right_apply_to_csr);
GKO_REGISTER_OPERATION(convert_to_csr, diagonal::convert_to_csr);
GKO_REGISTER_OPERATION(conj_transpose, diagonal::conj_transpose);


}  // namespace diagonal


template <typename ValueType, typename IndexType>
void Diagonal<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();

    if (dynamic_cast<const Dense<ValueType> *>(b) &&
        dynamic_cast<Dense<ValueType> *>(x)) {
        exec->run(diagonal::make_apply_to_dense(this, as<Dense<ValueType>>(b),
                                                as<Dense<ValueType>>(x)));
    } else if (dynamic_cast<const Csr<ValueType, IndexType> *>(b) &&
               dynamic_cast<Csr<ValueType, IndexType> *>(x)) {
        exec->run(
            diagonal::make_apply_to_csr(this, as<Csr<ValueType, IndexType>>(b),
                                        as<Csr<ValueType, IndexType>>(x)));
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void Diagonal<ValueType, IndexType>::rapply_impl(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();

    if (dynamic_cast<const Dense<ValueType> *>(b) &&
        dynamic_cast<Dense<ValueType> *>(x)) {
        exec->run(diagonal::make_right_apply_to_dense(
            this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(x)));
    } else if (dynamic_cast<const Csr<ValueType, IndexType> *>(b) &&
               dynamic_cast<Csr<ValueType, IndexType> *>(x)) {
        exec->run(diagonal::make_right_apply_to_csr(
            this, as<Csr<ValueType, IndexType>>(b),
            as<Csr<ValueType, IndexType>>(x)));
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void Diagonal<ValueType, IndexType>::apply_impl(const LinOp *alpha,
                                                const LinOp *b,
                                                const LinOp *beta,
                                                LinOp *x) const
{
    if (dynamic_cast<const Dense<ValueType> *>(b) &&
        dynamic_cast<Dense<ValueType> *>(x)) {
        auto dense_x = as<Dense<ValueType>>(x);
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, b);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Diagonal<ValueType, IndexType>::transpose() const
{
    return this->clone();
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Diagonal<ValueType, IndexType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto tmp =
        Diagonal<ValueType, IndexType>::create(exec, this->get_size()[0]);

    exec->run(diagonal::make_conj_transpose(this, tmp.get()));
    return std::move(tmp);
}


template <typename ValueType, typename IndexType>
void Diagonal<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Csr<ValueType, IndexType>::create(exec, this->get_size(),
                                                 this->get_size()[0]);
    exec->run(diagonal::make_convert_to_csr(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Diagonal<ValueType, IndexType>::move_to(Csr<ValueType, IndexType> *result)
{
    this->convert_to(result);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType *mtx, MatrixData &data)
{
    std::unique_ptr<const LinOp> op{};
    const MatrixType *tmp{};
    if (mtx->get_executor()->get_master() != mtx->get_executor()) {
        op = mtx->clone(mtx->get_executor()->get_master());
        tmp = static_cast<const MatrixType *>(op.get());
    } else {
        tmp = mtx;
    }

    data = {mtx->get_size(), {}};
    const auto values = mtx->get_const_values();

    for (size_type row = 0; row < data.size[0]; ++row) {
        data.nonzeros.emplace_back(row, row, values[row]);
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void Diagonal<ValueType, IndexType>::write(mat_data &data) const
{
    write_impl(this, data);
}


template <typename ValueType, typename IndexType>
void Diagonal<ValueType, IndexType>::write(mat_data32 &data) const
{
    write_impl(this, data);
}


#define GKO_DECLARE_DIAGONAL_MATRIX(value_type, index_type) \
    class Diagonal<value_type, index_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIAGONAL_MATRIX);


}  // namespace matrix
}  // namespace gko