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

#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/multigrid/amgx_pgm_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class AmgxPgm : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using RestrictProlong = gko::multigrid::AmgxPgm<value_type, index_type>;
    using T = value_type;
    using rmc_value_type = gko::remove_complex<value_type>;
    using WeightMtx = gko::matrix::Csr<rmc_value_type, index_type>;
    AmgxPgm()
        : exec(gko::ReferenceExecutor::create()),
          amgxpgm_factory(RestrictProlong::build()
                              .with_max_iterations(2u)
                              .with_max_unassigned_percentage(0.1)
                              .on(exec)),
          fine_b(gko::initialize<Vec>(
              {I<T>({2.0, -1.0}), I<T>({-1.0, 2.0}), I<T>({0.0, -1.0}),
               I<T>({3.0, -2.0}), I<T>({-2.0, 1.0})},
              exec)),
          coarse_b(gko::initialize<Vec>({I<T>({2.0, -1.0}), I<T>({0.0, -1.0})},
                                        exec)),
          restrict_ans((gko::initialize<Vec>(
              {I<T>({0.0, -1.0}), I<T>({2.0, 0.0})}, exec))),
          prolong_ans(gko::initialize<Vec>(
              {I<T>({0.0, -2.0}), I<T>({1.0, -2.0}), I<T>({1.0, -2.0}),
               I<T>({0.0, -1.0}), I<T>({2.0, 1.0})},
              exec)),
          fine_x(gko::initialize<Vec>(
              {I<T>({-2.0, -1.0}), I<T>({1.0, -1.0}), I<T>({-1.0, -1.0}),
               I<T>({0.0, 0.0}), I<T>({0.0, 2.0})},
              exec)),
          mtx(Mtx::create(exec, gko::dim<2>(5, 5), 15,
                          std::make_shared<typename Mtx::classical>())),
          weight(WeightMtx::create(
              exec, gko::dim<2>(5, 5), 15,
              std::make_shared<typename WeightMtx::classical>())),
          coarse(Mtx::create(exec, gko::dim<2>(2, 2), 4,
                             std::make_shared<typename Mtx::classical>())),
          mtx_diag(exec, 5),
          agg(exec, 5)
    {
        this->create_mtx(mtx.get(), weight.get(), &mtx_diag, &agg,
                         coarse.get());
        rstr_prlg = amgxpgm_factory->generate(mtx);
    }

    void create_mtx(Mtx *fine, WeightMtx *weight,
                    gko::Array<rmc_value_type> *diag,
                    gko::Array<index_type> *agg, Mtx *coarse)
    {
        auto vals = fine->get_values();
        auto cols = fine->get_col_idxs();
        auto rows = fine->get_row_ptrs();
        auto w_vals = weight->get_values();
        auto w_cols = weight->get_col_idxs();
        auto w_rows = weight->get_row_ptrs();
        auto diag_val = diag->get_data();
        auto agg_val = agg->get_data();
        auto c_vals = coarse->get_values();
        auto c_cols = coarse->get_col_idxs();
        auto c_rows = coarse->get_row_ptrs();
        /* this matrix is stored:
         *  5 -3 -3  0  0
         * -3  5  0 -2 -1
         * -3  0  5  0 -1
         *  0 -3  0  5  0
         *  0 -2 -2  0  5
         */
        vals[0] = 5;
        vals[1] = -3;
        vals[2] = -3;
        vals[3] = -3;
        vals[4] = 5;
        vals[5] = -2;
        vals[6] = -1;
        vals[7] = -3;
        vals[8] = 5;
        vals[9] = -1;
        vals[10] = -3;
        vals[11] = 5;
        vals[12] = -2;
        vals[13] = -2;
        vals[14] = 5;

        rows[0] = 0;
        rows[1] = 3;
        rows[2] = 7;
        rows[3] = 10;
        rows[4] = 12;
        rows[5] = 15;

        cols[0] = 0;
        cols[1] = 1;
        cols[2] = 2;
        cols[3] = 0;
        cols[4] = 1;
        cols[5] = 3;
        cols[6] = 4;
        cols[7] = 0;
        cols[8] = 2;
        cols[9] = 4;
        cols[10] = 1;
        cols[11] = 3;
        cols[12] = 1;
        cols[13] = 2;
        cols[14] = 4;

        /* weight matrix is stored:
         * 5   3   3   0   0
         * 3   5   0   2.5 1.5
         * 3   0   5   0   1.5
         * 0   2.5 0   5   0
         * 0   1.5 1.5 0   5
         */
        w_vals[0] = 5;
        w_vals[1] = 3;
        w_vals[2] = 3;
        w_vals[3] = 3;
        w_vals[4] = 5;
        w_vals[5] = 2.5;
        w_vals[6] = 1.5;
        w_vals[7] = 3;
        w_vals[8] = 5;
        w_vals[9] = 1.5;
        w_vals[10] = 2.5;
        w_vals[11] = 5;
        w_vals[12] = 1.5;
        w_vals[13] = 1.5;
        w_vals[14] = 5;

        w_rows[0] = 0;
        w_rows[1] = 3;
        w_rows[2] = 7;
        w_rows[3] = 10;
        w_rows[4] = 12;
        w_rows[5] = 15;

        w_cols[0] = 0;
        w_cols[1] = 1;
        w_cols[2] = 2;
        w_cols[3] = 0;
        w_cols[4] = 1;
        w_cols[5] = 3;
        w_cols[6] = 4;
        w_cols[7] = 0;
        w_cols[8] = 2;
        w_cols[9] = 4;
        w_cols[10] = 1;
        w_cols[11] = 3;
        w_cols[12] = 1;
        w_cols[13] = 2;
        w_cols[14] = 4;

        diag_val[0] = 5;
        diag_val[1] = 5;
        diag_val[2] = 5;
        diag_val[3] = 5;
        diag_val[4] = 5;

        agg_val[0] = 0;
        agg_val[1] = 1;
        agg_val[2] = 0;
        agg_val[3] = 1;
        agg_val[4] = 0;

        /* this coarse is stored:
         *  6 -5
         * -4  5
         */
        c_vals[0] = 6;
        c_vals[1] = -5;
        c_vals[2] = -4;
        c_vals[3] = 5;

        c_rows[0] = 0;
        c_rows[1] = 2;
        c_rows[2] = 4;

        c_cols[0] = 0;
        c_cols[1] = 1;
        c_cols[2] = 0;
        c_cols[3] = 1;
    }

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        ASSERT_EQ(m1->get_num_stored_elements(), m2->get_num_stored_elements());
        for (gko::size_type i = 0; i < m1->get_size() + 1; i++) {
            ASSERT_EQ(m1->get_const_row_ptrs()[i], m2->get_const_row_ptrs()[i]);
        }
        for (gko::size_type i = 0; i < m1->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m1->get_const_values()[i], m2->get_const_values()[i]);
            EXPECT_EQ(m1->get_const_col_idxs()[i], m2->get_const_col_idxs()[i]);
        }
    }

    static void assert_same_agg(const index_type *m1, const index_type *m2,
                                gko::size_type len)
    {
        for (gko::size_type i = 0; i < len; ++i) {
            EXPECT_EQ(m1[i], m2[i]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> coarse;
    std::shared_ptr<WeightMtx> weight;
    gko::Array<rmc_value_type> mtx_diag;
    gko::Array<index_type> agg;
    std::shared_ptr<Vec> coarse_b;
    std::shared_ptr<Vec> fine_b;
    std::shared_ptr<Vec> restrict_ans;
    std::shared_ptr<Vec> prolong_ans;
    std::shared_ptr<Vec> fine_x;
    std::unique_ptr<typename RestrictProlong::Factory> amgxpgm_factory;
    std::unique_ptr<RestrictProlong> rstr_prlg;
};

TYPED_TEST_CASE(AmgxPgm, gko::test::ValueIndexTypes);


TYPED_TEST(AmgxPgm, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using RestrictProlong = typename TestFixture::RestrictProlong;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->rstr_prlg.get());

    auto copy_mtx =
        static_cast<RestrictProlong *>(copy.get())->get_system_matrix();
    auto copy_agg = static_cast<RestrictProlong *>(copy.get())->get_const_agg();
    auto copy_coarse = copy->get_coarse_operator();

    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(copy_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(copy_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using RestrictProlong = typename TestFixture::RestrictProlong;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->rstr_prlg));

    auto copy_mtx =
        static_cast<RestrictProlong *>(copy.get())->get_system_matrix();
    auto copy_agg = static_cast<RestrictProlong *>(copy.get())->get_const_agg();
    auto copy_coarse = copy->get_coarse_operator();

    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(copy_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(copy_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using RestrictProlong = typename TestFixture::RestrictProlong;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    auto clone = this->rstr_prlg->clone();

    auto clone_mtx =
        static_cast<RestrictProlong *>(clone.get())->get_system_matrix();
    auto clone_agg =
        static_cast<RestrictProlong *>(clone.get())->get_const_agg();
    auto clone_coarse = clone->get_coarse_operator();

    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(clone_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(clone_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeCleared)
{
    using RestrictProlong = typename TestFixture::RestrictProlong;

    this->rstr_prlg->clear();

    auto mtx = static_cast<RestrictProlong *>(this->rstr_prlg.get())
                   ->get_system_matrix();
    auto coarse = this->rstr_prlg->get_coarse_operator();
    auto agg = static_cast<RestrictProlong *>(this->rstr_prlg.get())->get_agg();

    ASSERT_EQ(mtx, nullptr);
    ASSERT_EQ(coarse, nullptr);
    ASSERT_EQ(agg, nullptr);
}


TYPED_TEST(AmgxPgm, RestrictApply)
{
    // fine->coarse
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = Vec::create_with_config_of(gko::lend(this->coarse_b));

    gko::kernels::reference::amgx_pgm::restrict_apply(
        this->exec, this->agg, this->fine_b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->restrict_ans,
                        gko::remove_complex<value_type>{0});
}

TYPED_TEST(AmgxPgm, ProlongApplyadd)
{
    using value_type = typename TestFixture::value_type;
    auto x = gko::clone(this->fine_x);

    gko::kernels::reference::amgx_pgm::prolong_applyadd(
        this->exec, this->agg, this->coarse_b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->prolong_ans,
                        gko::remove_complex<value_type>{0});
}

TYPED_TEST(AmgxPgm, MatchEdge)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    gko::Array<index_type> snb(this->exec, 5);
    auto agg_val = agg.get_data();
    auto snb_val = snb.get_data();
    for (int i = 0; i < 5; i++) {
        agg_val[i] = -1;
    }
    snb_val[0] = 2;
    snb_val[1] = 0;
    snb_val[2] = 0;
    snb_val[3] = 1;
    snb_val[4] = 2;

    gko::kernels::reference::amgx_pgm::match_edge(this->exec, snb, agg);

    ASSERT_EQ(agg_val[0], 0);
    ASSERT_EQ(agg_val[1], -1);
    ASSERT_EQ(agg_val[2], 0);
    ASSERT_EQ(agg_val[3], -1);
    ASSERT_EQ(agg_val[4], -1);
}

TYPED_TEST(AmgxPgm, CountUnagg)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    auto agg_val = agg.get_data();
    gko::size_type num_unagg = 0;
    agg_val[0] = 0;
    agg_val[1] = -1;
    agg_val[2] = 0;
    agg_val[3] = -1;
    agg_val[4] = -1;

    gko::kernels::reference::amgx_pgm::count_unagg(this->exec, agg, &num_unagg);

    ASSERT_EQ(num_unagg, 3);
}


TYPED_TEST(AmgxPgm, Renumber)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    auto agg_val = agg.get_data();
    gko::size_type num_agg = 0;
    agg_val[0] = 0;
    agg_val[1] = 1;
    agg_val[2] = 0;
    agg_val[3] = 1;
    agg_val[4] = 4;

    gko::kernels::reference::amgx_pgm::renumber(this->exec, agg, &num_agg);

    ASSERT_EQ(num_agg, 3);
    ASSERT_EQ(agg_val[0], 0);
    ASSERT_EQ(agg_val[1], 1);
    ASSERT_EQ(agg_val[2], 0);
    ASSERT_EQ(agg_val[3], 1);
    ASSERT_EQ(agg_val[4], 2);
}


TYPED_TEST(AmgxPgm, Generate)
{
    auto coarse_fine = this->amgxpgm_factory->generate(this->mtx);

    auto agg_result = coarse_fine->get_const_agg();

    ASSERT_EQ(agg_result[0], 0);
    ASSERT_EQ(agg_result[1], 1);
    ASSERT_EQ(agg_result[2], 0);
    ASSERT_EQ(agg_result[3], 1);
    ASSERT_EQ(agg_result[4], 0);
}


TYPED_TEST(AmgxPgm, CoarseFineRestrictApply)
{
    std::unique_ptr<gko::multigrid::RestrictProlong> amgx_pgm{
        this->amgxpgm_factory->generate(this->mtx)};

    // fine->coarse
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = Vec::create_with_config_of(gko::lend(this->coarse_b));
    amgx_pgm->restrict_apply(this->fine_b.get(), x.get());
    GKO_ASSERT_MTX_NEAR(x, this->restrict_ans,
                        gko::remove_complex<value_type>{0});
}


TYPED_TEST(AmgxPgm, CoarseFineProlongApplyadd)
{
    using value_type = typename TestFixture::value_type;
    std::unique_ptr<gko::multigrid::RestrictProlong> amgx_pgm{
        this->amgxpgm_factory->generate(this->mtx)};
    auto x = gko::clone(this->fine_x);

    amgx_pgm->prolong_applyadd(this->coarse_b.get(), x.get());
    GKO_ASSERT_MTX_NEAR(x, this->prolong_ans,
                        gko::remove_complex<value_type>{0});
}


TYPED_TEST(AmgxPgm, ExtractDiag)
{
    using rmc_value_type = typename TestFixture::rmc_value_type;
    gko::Array<rmc_value_type> diag(this->exec, 5);

    gko::kernels::reference::amgx_pgm::extract_diag(this->exec,
                                                    this->weight.get(), diag);
    GKO_ASSERT_ARRAY_EQ(diag, this->mtx_diag);
}


TYPED_TEST(AmgxPgm, FindStrongestNeighbor)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> strongest_neighbor(this->exec, 5);
    gko::Array<index_type> agg(this->exec, 5);
    auto snb_vals = strongest_neighbor.get_data();
    auto agg_vals = agg.get_data();
    for (int i = 0; i < 5; i++) {
        snb_vals[i] = -1;
        agg_vals[i] = -1;
    }

    gko::kernels::reference::amgx_pgm::find_strongest_neighbor(
        this->exec, this->weight.get(), this->mtx_diag, agg,
        strongest_neighbor);

    ASSERT_EQ(snb_vals[0], 2);
    ASSERT_EQ(snb_vals[1], 0);
    ASSERT_EQ(snb_vals[2], 0);
    ASSERT_EQ(snb_vals[3], 1);
    ASSERT_EQ(snb_vals[4], 2);
}


TYPED_TEST(AmgxPgm, AssignToExistAgg)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    gko::Array<index_type> intermediate_agg(this->exec, 0);
    auto agg_vals = agg.get_data();
    // 0 - 2, 1 - 3
    agg_vals[0] = 0;
    agg_vals[1] = 1;
    agg_vals[2] = 0;
    agg_vals[3] = 1;
    agg_vals[4] = -1;

    gko::kernels::reference::amgx_pgm::assign_to_exist_agg(
        this->exec, this->weight.get(), this->mtx_diag, agg, intermediate_agg);

    ASSERT_EQ(agg_vals[0], 0);
    ASSERT_EQ(agg_vals[1], 1);
    ASSERT_EQ(agg_vals[2], 0);
    ASSERT_EQ(agg_vals[3], 1);
    ASSERT_EQ(agg_vals[4], 0);
}


TYPED_TEST(AmgxPgm, GenerateMtx)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using mtx_type = typename TestFixture::Mtx;
    gko::Array<index_type> agg(this->exec, 5);
    auto agg_vals = agg.get_data();
    // 0 - 2, 1 - 3, 4
    agg_vals[0] = 0;
    agg_vals[1] = 1;
    agg_vals[2] = 0;
    agg_vals[3] = 1;
    agg_vals[4] = 2;
    auto csr_coarse = mtx_type::create(this->exec, gko::dim<2>{3, 3}, 0);

    gko::kernels::reference::amgx_pgm::amgx_pgm_generate(
        this->exec, this->mtx.get(), agg, csr_coarse.get());

    auto r = csr_coarse->get_const_row_ptrs();
    auto c = csr_coarse->get_const_col_idxs();
    auto v = csr_coarse->get_const_values();
    ASSERT_EQ(csr_coarse->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(csr_coarse->get_num_stored_elements(), 9);
    ASSERT_EQ(r[0], 0);
    ASSERT_EQ(r[1], 3);
    ASSERT_EQ(r[2], 6);
    ASSERT_EQ(r[3], 9);
    ASSERT_EQ(c[0], 0);
    ASSERT_EQ(c[1], 1);
    ASSERT_EQ(c[2], 2);
    ASSERT_EQ(c[3], 0);
    ASSERT_EQ(c[4], 1);
    ASSERT_EQ(c[5], 2);
    ASSERT_EQ(c[6], 0);
    ASSERT_EQ(c[7], 1);
    ASSERT_EQ(c[8], 2);
    ASSERT_EQ(v[0], value_type{4});
    ASSERT_EQ(v[1], value_type{-3});
    ASSERT_EQ(v[2], value_type{-1});
    ASSERT_EQ(v[3], value_type{-3});
    ASSERT_EQ(v[4], value_type{5});
    ASSERT_EQ(v[5], value_type{-1});
    ASSERT_EQ(v[6], value_type{-2});
    ASSERT_EQ(v[7], value_type{-2});
    ASSERT_EQ(v[8], value_type{5});
}


}  // namespace
