/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file inplace_merge.h
 *  \brief Sequential implementation of inplace_merge algorithms.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <cuda/std/__algorithm/move.h>
#include <cuda/std/__utility/move.h>


THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
namespace inplace_merge_detail
{
template <typename Predicate>
class invert // invert the sense of a comparison
{
private:
  Predicate p_{};

public:
  invert() = default;

  explicit invert(Predicate p)
      : p_(p)
  {}

  template <class T1>
  bool operator()(const T1& x)
  {
    return !p_(x);
  }

  template <class T1, class T2>
  bool operator()(const T1& x, const T2& y)
  {
    return p_(y, x);
  }
};

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
void half_inplace_merge(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  for (; first1 != last1; ++result)
  {
    if (first2 == last2)
    {
      ::cuda::std::move(first1, last1, result);
      return;
    }

    if (comp(*first2, *first1))
    {
      *result = ::cuda::std::move(*first2);
      ++first2;
    }
    else
    {
      *result = ::cuda::std::move(*first1);
      ++first1;
    }
  }
  // first2 through last2 are already in the right spot.
}
} // namespace inplace_merge_detail

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void inplace_merge(
  sequential::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator middle,
  InputIterator last,
  StrictWeakOrdering comp)
{
  using value_type      = thrust::detail::it_value_t<InputIterator>;
  using difference_type = thrust::detail::it_difference_t<InputIterator>;

  difference_type len1 = ::cuda::std::distance(first, middle);
  difference_type len2 = ::cuda::std::distance(middle, last);

  if (len1 > len2)
  {
    thrust::detail::temporary_array<value_type, DerivedPolicy> buf(exec, first, middle);
    inplace_merge_detail::half_inplace_merge(buf.begin(), buf.end(), middle, last, first, comp);
  }
  else
  {
    thrust::detail::temporary_array<value_type, DerivedPolicy> buf(exec, middle, last);
    
    using buf_iter = typename thrust::detail::temporary_array<value_type, DerivedPolicy>::iterator;
    thrust::reverse_iterator<buf_iter> r_buf_first(buf.end());
    thrust::reverse_iterator<buf_iter> r_buf_last(buf.begin());

    thrust::reverse_iterator<InputIterator> r_first(last);
    thrust::reverse_iterator<InputIterator> r_middle(middle);
    thrust::reverse_iterator<InputIterator> r_last(first);

    // thrust::system::detail::sequential::merge(exec, r_buf_first, r_buf_last, r_middle, r_last, r_first, inv_comp);
    inplace_merge_detail::half_inplace_merge(
      r_buf_first, r_buf_last, r_middle, r_last, r_first, inplace_merge_detail::invert(comp));
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void inplace_merge_by_key(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 keys_first,
  InputIterator1 keys_middle,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  StrictWeakOrdering comp)
{
  using iterator_tuple = thrust::tuple<InputIterator1, InputIterator2>;
  using zip_iterator   = thrust::zip_iterator<iterator_tuple>;

  InputIterator2 values_middle = values_first + (keys_middle - keys_first);
  InputIterator2 values_last   = values_first + (keys_last - keys_first);

  zip_iterator zipped_first  = thrust::make_zip_iterator(keys_first, values_first);
  zip_iterator zipped_middle = thrust::make_zip_iterator(keys_middle, values_middle);
  zip_iterator zipped_last   = thrust::make_zip_iterator(keys_last, values_last);
  thrust::detail::compare_first<StrictWeakOrdering> comp_first{comp};

  inplace_merge(sequential::execution_policy<DerivedPolicy>{}, zipped_first, zipped_middle, zipped_last, comp_first);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
