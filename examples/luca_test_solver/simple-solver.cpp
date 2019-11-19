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

#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


constexpr std::size_t number_of_rhs{100};

using dense = gko::matrix::Dense<>;
using csr = gko::matrix::Csr<>;
using coo = gko::matrix::Coo<>;
using bj = gko::preconditioner::Jacobi<>;


template <typename Solver, typename ExecType>
std::unique_ptr<typename Solver::Factory> generate_solver_factory(
    const ExecType &exec, bool with_preconditioner)
{
    constexpr unsigned int max_iter{1000};
    constexpr double red_factor{1e-15};
    if (with_preconditioner) {
        return Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_iter).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(red_factor)
                    .on(exec))
            .with_preconditioner(bj::build().with_max_block_size(32u).on(exec))
            .on(exec);
    } else {
        return Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_iter).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(red_factor)
                    .on(exec))
            .on(exec);
    }
}


template <typename Solver, typename MatrixFormat, typename ExecType>
void create_and_run_solver(const ExecType &exec, bool with_preconditioner,
                           const std::unique_ptr<dense> &zero,
                           const std::unique_ptr<dense> &neg_one,
                           const std::shared_ptr<MatrixFormat> &Mi,
                           const std::unique_ptr<dense> &b)
{
    auto delta_vm =
        dense::create(exec, gko::dim<2>{Mi->get_size()[1], b->get_size()[1]});
    // Initial guess: zero vector
    delta_vm->scale(zero.get());

    auto solver_gen =
        generate_solver_factory<Solver>(exec, with_preconditioner);
    auto solver = solver_gen->generate(Mi);

    std::shared_ptr<gko::log::Stream<>> stream_logger =
        gko::log::Stream<>::create(
            exec, gko::log::Logger::iteration_complete_mask, std::cout);
    solver->add_logger(stream_logger);

    std::cout << "With";
    if (!with_preconditioner) {
        std::cout << "out";
    }
    std::cout << " preconditioner:\n";

    auto start_solver = std::chrono::steady_clock::now();
    solver->apply(lend(b), lend(delta_vm));
    auto end_solver = std::chrono::steady_clock::now();

    auto dur_solver = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_solver - start_solver)
                          .count();

    std::cout << "Time for solving [us]: " << dur_solver << std::endl;

    // std::cout << "Solution (x): \n";
    // write(std::cout, lend(x));

    auto diff_vector = dense::create_with_config_of(lend(delta_vm));
    auto res = dense::create(exec, gko::dim<2>{1, delta_vm->get_size()[1]});
    Mi->apply(lend(delta_vm), lend(diff_vector));
    diff_vector->add_scaled(lend(neg_one), lend(b));
    diff_vector->compute_norm2(lend(res));

    auto ref_exec = gko::ReferenceExecutor::create();
    auto res_ref = dense::create(ref_exec);
    res_ref->copy_from(lend(res));


    std::cout << "Residual norm sqrt(r^T r): \n";
    write(std::cout, lend(res_ref));
}


int main(int argc, char *argv[])
{
    //*
    auto strategy = std::make_shared<csr::automatical>();
    /*/
    // auto strategy = std::make_shared<csr::load_balance>();
    auto strategy = std::make_shared<csr::classical>();
    //*/

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    } else if (argc == 2 && std::string(argv[1]) == "hip" &&
               gko::HipExecutor::get_num_devices() > 0) {
        exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }
    std::shared_ptr<gko::Executor> host_exec = exec->get_master();

    std::string location_matrices{
        /*
        "/home/thomas/projects/matrices/Downloaded/Matrices_Luca_Azzolin/"};
        /*/
        "/home/thoasm/projects/matrices/luca_matrices/Matrices_Luca_Azzolin/"};
    //*/
    std::vector<std::string> location_Ki = {
        location_matrices + "Reentry/Ki_reentries.mtx",
        location_matrices + "Repolarization_depolarization/Ki_one_beat.mtx",
        location_matrices + "Silence/Ki_repolarization.mtx"};
    std::vector<std::string> location_Mi = {
        location_matrices + "Reentry/Mi_reentries.mtx",
        location_matrices + "Repolarization_depolarization/Mi_one_beat.mtx",
        location_matrices + "Silence/Mi_repolarization.mtx"};
    std::vector<std::string> location_vm = {
        location_matrices + "Reentry/vm_reentry.mtx",
        location_matrices + "Repolarization_depolarization/act_one_beat.mtx",
        // location_matrices + "Repolarization_depolarization/rep_one_beat.mtx",
        location_matrices + "Silence/vm_repolarization.mtx"};

    auto one = gko::initialize<dense>({1.0}, exec);
    auto neg_one = gko::initialize<dense>({-1.0}, exec);
    auto zero = gko::initialize<dense>({0.0}, exec);

    for (std::size_t i = 0; i < location_Ki.size(); ++i) {
        std::cout << "\nLoading Matrices from: "
                  << location_Ki[i].substr(0, location_Ki[i].find_last_of('/'))
                  << "\n\n"
                  << std::flush;
        auto Ki = gko::read<csr>(std::ifstream(location_Ki[i]), exec, strategy);
        auto Mi = gko::share(
            gko::read<csr>(std::ifstream(location_Mi[i]), exec, strategy));
        auto vm_file =
            gko::read<dense>(std::ifstream(location_vm[i]), host_exec);
        auto vm_new = dense::create(
            host_exec, gko::dim<2>{vm_file->get_size()[0], number_of_rhs});
        auto vm = dense::create(exec);
        auto b = dense::create(
            exec, gko::dim<2>(Ki->get_size()[0], vm_new->get_size()[1]));

        // Copy and duplicate the vector from file to have the desired number of
        // right hand sides.
        std::size_t num_rows{vm_file->get_size()[0]};
        std::size_t old_num_rhs{vm_file->get_size()[1]};
        std::size_t new_num_rhs{vm_new->get_size()[1]};
        for (std::size_t i = 0; i < vm_new->get_size()[0]; ++i) {
            std::size_t old_offset{i * old_num_rhs};
            std::size_t new_offset{i * new_num_rhs};
            for (std::size_t j = 0; j < vm_new->get_size()[1]; ++j) {
                vm_new->get_values()[new_offset + j] =
                    vm_file->get_values()[old_offset + j % old_num_rhs];
            }
        }

        vm_new->move_to(vm.get());

        Ki->apply(lend(vm), lend(b));
        b->scale(lend(neg_one));

        using Bicgstab = gko::solver::Bicgstab<>;
        using Cg = gko::solver::Cg<>;
        using Cgs = gko::solver::Cgs<>;
        using Fcg = gko::solver::Fcg<>;
        using Gmres = gko::solver::Gmres<>;

        /*
        std::cout << "\n\nUsing BiCGSTAB for solving:\n\n" << std::flush;
        create_and_run_solver<Bicgstab>(exec, false, zero, neg_one, Mi, b);
        create_and_run_solver<Bicgstab>(exec, true, zero, neg_one, Mi, b);
        //*/

        std::cout << "\n\nUsing CG for solving:\n\n" << std::flush;
        create_and_run_solver<Cg>(exec, false, zero, neg_one, Mi, b);
        create_and_run_solver<Cg>(exec, true, zero, neg_one, Mi, b);

        /*
        std::cout << "\n\nUsing CGS for solving:\n\n" << std::flush;
        create_and_run_solver<Cgs>(exec, false, zero, neg_one, Mi, b);
        create_and_run_solver<Cgs>(exec, true, zero, neg_one, Mi, b);

        std::cout << "\n\nUsing FCG for solving:\n\n" << std::flush;
        create_and_run_solver<Fcg>(exec, false, zero, neg_one, Mi, b);
        create_and_run_solver<Fcg>(exec, true, zero, neg_one, Mi, b);

        std::cout << "\n\nUsing GMRES for solving:\n\n" << std::flush;
        create_and_run_solver<Gmres>(exec, false, zero, neg_one, Mi, b);
        create_and_run_solver<Gmres>(exec, true, zero, neg_one, Mi, b);
        //*/
    }
}
