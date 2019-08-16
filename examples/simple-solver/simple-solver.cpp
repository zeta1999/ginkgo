/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <chrono>
#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <string>

#define USE_SINGLE 1

constexpr size_t runs_per_iter{4};

int main(int argc, char *argv[])
{
    using vec = gko::matrix::Dense<>;
    using mtx = gko::matrix::Csr<>;

    using bicgstab = gko::solver::Bicgstab<>;
    using cg = gko::solver::Cg<>;
    using cgs = gko::solver::Cgs<>;
    using gmres = gko::solver::Gmres<>;

#if USE_SINGLE
    constexpr size_t runs_per_exec{runs_per_iter};
    constexpr size_t number_execs{1};
#else
    constexpr size_t runs_per_exec{1};
    constexpr size_t number_execs{runs_per_iter};
#endif

    constexpr size_t iter_num{20};
    constexpr unsigned int solver_max_iter{10u};
    constexpr double solver_max_res{1e-15};
    std::string mtx_file1 =
        "/home/thoasm/projects/ginkgo_github/matrices/test/ani4.mtx";
    std::vector<double> vec_x1(3081, 1.0);
    std::vector<double> vec_b1(3081, 2.0);

    auto exec_omp = gko::OmpExecutor::create();

    std::vector<std::shared_ptr<gko::Executor>> exec(number_execs);
    std::vector<std::shared_ptr<mtx>> A(number_execs);
    std::vector<std::shared_ptr<vec>> b(number_execs);
    std::vector<std::shared_ptr<vec>> x(number_execs);
    std::vector<std::shared_ptr<gko::LinOp>> solver(number_execs);

    for (size_t i = 0; i < number_execs; ++i) {
        exec[i] = gko::CudaExecutor::create(1, exec_omp);
        A[i] = share(gko::read<mtx>(std::ifstream(mtx_file1), exec[i]));

        auto b_array =
            gko::Array<double>(exec[i], vec_b1.begin(), vec_b1.end());
        b[i] =
            vec::create(exec[i], gko::dim<2>(3081, 1), std::move(b_array), 1);
        auto x_array =
            gko::Array<double>(exec[i], vec_x1.begin(), vec_x1.end());
        x[i] =
            vec::create(exec[i], gko::dim<2>(3081, 1), std::move(x_array), 1);
        auto solver_gen =
            cg::build()
                .with_criteria(gko::stop::Iteration::build()
                                   .with_max_iters(solver_max_iter)
                                   .on(exec[i]),
                               gko::stop::ResidualNormReduction<>::build()
                                   .with_reduction_factor(solver_max_res)
                                   .on(exec[i]))
                .on(exec[i]);
        solver[i] = solver_gen->generate(A[i]);
    }

    for (size_t i = 0; i < exec.size(); ++i) {
        exec[i]->synchronize();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iter_num; ++i) {
        for (size_t r = 0; r < runs_per_exec; ++r) {
            for (size_t ne = 0; ne < exec.size(); ++ne) {
                solver[ne]->apply(gko::lend(b[ne]), gko::lend(x[ne]));
            }
        }
    }
    for (size_t i = 0; i < exec.size(); ++i) {
        exec[i]->synchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count();
    std::cout << "Time needed (on average) for " << number_execs
              << " executor: " << dur / iter_num << " ns" << std::endl;
}
