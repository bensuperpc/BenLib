/**
 * @file repository.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef BENLIB_PATERN_REPOSITORY_HPP_
#define BENLIB_PATERN_REPOSITORY_HPP_

#include <list>

namespace benlib {
namespace pattern {

template <typename parentType>
class Repository {
   private:
    static inline std::list<parentType*> _data;

   protected:
    Repository(const Repository&) = delete;
    Repository& operator=(const Repository&) = delete;

    explicit Repository() {
        static_assert(std::is_base_of_v<Repository<parentType>, parentType>, "parentType must inherit from Repository<parentType>");
        Repository<parentType>::_data.push_back(static_cast<parentType*>(this));
    }

   public:
    virtual ~Repository() {
        Repository<parentType>::_data.remove(static_cast<parentType*>(this)); 
    }
    static inline std::list<parentType*>& data() { return Repository<parentType>::_data; }
    static inline void clear() { Repository<parentType>::_data.clear(); }
};

}  // namespace pattern
}  // namespace benlib
#endif
