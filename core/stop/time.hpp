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

#ifndef GKO_CORE_STOP_TIME_HPP_
#define GKO_CORE_STOP_TIME_HPP_


#include "core/stop/criterion.hpp"


#include <chrono>


namespace gko {
namespace stop {

/**
 * The Time class is a stopping criterion which stops the iteration process
 * after a certain amout of time has passed.
 */
class Time : public EnablePolymorphicObject<Time, Criterion> {
    friend class EnablePolymorphicObject<Time, Criterion>;

public:
    using clock = std::chrono::system_clock;

    bool check(uint8 stoppingId, bool setFinalized,
               Array<stopping_status> *stop_status, bool *one_changed,
               const Updater &) override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Amount of seconds to wait
         */
        double GKO_FACTORY_PARAMETER(time_limit, 10.0);
    };
    GKO_ENABLE_CRITERION_FACTORY(Time, parameters, Factory);

protected:
    explicit Time(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<Time, Criterion>(std::move(exec))
    {}

    explicit Time(const Factory *factory, const CriterionArgs *args)
        : EnablePolymorphicObject<Time, Criterion>(factory->get_executor()),
          parameters_{factory->get_parameters()},
          time_limit_{std::chrono::duration<double>(
              factory->get_parameters().time_limit)},
          start_{clock::now()}
    {}

private:
    std::chrono::duration<double> time_limit_{};
    clock::time_point start_{};
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_TIME_HPP_
