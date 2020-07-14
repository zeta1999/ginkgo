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

// force-top: on
// TODO remove when the HIP includes are fixed
#include <hip/hip_runtime.h>
// force-top: off


#include "core/components/sqrt_array.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils/assertions.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class SqrtArray : public ::testing::Test {
protected:
    using value_type = double;
    SqrtArray()
        : ref(gko::ReferenceExecutor::create()),
          exec(gko::HipExecutor::create(0, ref)),
          total_size(6344),
          vals(ref, total_size),
          rsqrt(ref, total_size),
          dsqrt(exec, total_size)
    {
        std::fill_n(vals.get_data(), total_size, 1234.0);
        std::fill_n(vals.get_data() + total_size - 10, 10, 163.0);
        std::fill_n(rsqrt.get_data(), total_size, std::sqrt(1234.0));
        std::fill_n(rsqrt.get_data() + total_size - 10, 10, std::sqrt(163.0));
        dvals = gko::Array<value_type>{exec, vals};
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> exec;
    gko::size_type total_size;
    gko::Array<value_type> vals;
    gko::Array<value_type> dvals;
    gko::Array<value_type> rsqrt;
    gko::Array<value_type> dsqrt;
};


TEST_F(SqrtArray, EqualsReference)
{
    gko::kernels::hip::components::sqrt_array(
        exec, total_size, dvals.get_data(), dsqrt.get_data());
    GKO_ASSERT_ARRAY_EQ(rsqrt, dsqrt);
}


}  // namespace
