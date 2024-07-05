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

#include "../common/concept.hpp"

namespace benlib {
namespace patern {

template <typename parentType>
class Repository {
   private:
    static inline std::list<parentType*> _data;

   public:
    explicit Repository() {
        static_assert(std::is_base_of<Repository, parentType>::value, "parentType must be a child of Repository");
        Repository<parentType>::_data.push_back(static_cast<parentType*>(this));
    }
    virtual ~Repository() { Repository<parentType>::_data.remove(static_cast<parentType*>(this)); }
    static inline std::list<parentType*>& getData() { return Repository<parentType>::_data; }
    static inline void clear() { Repository<parentType>::_data.clear(); }
};

}  // namespace patern
}  // namespace benlib
#endif
