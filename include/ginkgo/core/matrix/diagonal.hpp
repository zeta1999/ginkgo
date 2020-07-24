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

#ifndef GKO_CORE_MATRIX_DIAGONAL_HPP_
#define GKO_CORE_MATRIX_DIAGONAL_HPP_


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


/**
 * This class is a utility which efficiently implements the diagonal matrix (a
 * linear operator which scales a vector row wise).
 *
 * Objects of the Diagonal class always represent a square matrix, and
 * require one array to store their values.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup diagonal
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Diagonal : public EnableLinOp<Diagonal<ValueType>>,
                 public EnableCreateMethod<Diagonal<ValueType>>,
                 public Transposable {
    friend class EnablePolymorphicObject<Diagonal, LinOp>;
    friend class EnableCreateMethod<Diagonal>;

public:
    using EnableLinOp<Diagonal>::convert_to;
    using EnableLinOp<Diagonal>::move_to;

    using value_type = ValueType;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Returns a pointer to the array of values of the matrix.
     *
     * @return the pointer to the array of values
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    void rapply(const LinOp *b, LinOp *x) const
    {
        GKO_ASSERT_EQ(this->get_size()[0], b->get_size()[1]);
        GKO_ASSERT_EQ(this->get_size()[1], x->get_size()[1]);
        GKO_ASSERT_EQ(b->get_size()[0], x->get_size()[0]);

        this->rapply_impl(b, x);
    }


protected:
    /**
     * Creates an empty Diagonal matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Diagonal(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Diagonal>(exec)
    {}

    /**
     * Creates an Diagonal matrix of the specified size.
     *
     * @param size  size of the matrix
     */
    Diagonal(std::shared_ptr<const Executor> exec, size_type size)
        : EnableLinOp<Diagonal>(exec, dim<2>{size}), values_(exec, size)
    {}

    /**
     * Creates a Diagonal matrix from an already allocated (and initialized)
     * array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray>
    Diagonal(std::shared_ptr<const Executor> exec, size_type size,
             ValuesArray &&values)
        : EnableLinOp<Diagonal>(exec, dim<2>{size}),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        GKO_ENSURE_IN_BOUNDS(size - 1, values_.get_num_elems());
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void rapply_impl(const LinOp *b, LinOp *x) const;


private:
    Array<value_type> values_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DIAGONAL_HPP_
