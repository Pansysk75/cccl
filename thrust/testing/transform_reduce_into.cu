#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/transform_reduce.h>

#include <unittest/unittest.h>

template <typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename OutputType,
          typename BinaryFunction>
void transform_reduce_into(
  my_system& system, InputIterator, InputIterator, OutputIterator, UnaryFunction, OutputType, BinaryFunction)
{
  system.validate_dispatch();
}

void TestTransformReduceIntoDispatchExplicit()
{
  thrust::device_vector<int> vec(1);
  thrust::device_vector<int> result(1);

  my_system sys(0);
  thrust::transform_reduce_into(sys, vec.begin(), vec.begin(), result.begin(), 0, 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTransformReduceIntoDispatchExplicit);

template <typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename OutputType,
          typename BinaryFunction>
void
transform_reduce_into(my_tag, InputIterator first, InputIterator, OutputIterator, UnaryFunction, OutputType, BinaryFunction)
{
  *first = 13;
}

void TestTransformReduceIntoDispatchImplicit()
{
  thrust::device_vector<int> vec(1);
  thrust::device_vector<int> result(1);

  thrust::transform_reduce_into(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), result.begin(), 0, 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformReduceIntoDispatchImplicit);

template <class Vector>
void TestTransformReduceIntoSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, -2, 3};
  Vector result(1);

  T init   = 10;
  thrust::transform_reduce_into(
    data.begin(), data.end(), result.begin(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());

  ASSERT_EQUAL(result[0], 8);
}
DECLARE_VECTOR_UNITTEST(TestTransformReduceIntoSimple);

template <typename T>
void TestTransformReduceInto(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T init = 13;
  thrust::host_vector<T> cpu_result(1);
  thrust::device_vector<T> gpu_result(1);

  thrust::transform_reduce_into(
    h_data.begin(), h_data.end(), cpu_result.begin(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());
  thrust::transform_reduce_into(
    d_data.begin(), d_data.end(), gpu_result.begin(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());

  ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduceInto);

template <typename T>
void TestTransformReduceIntoFromConst(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T init = 13;
  thrust::host_vector<T> cpu_result(1);
  thrust::device_vector<T> gpu_result(1);

  thrust::transform_reduce_into(
    h_data.cbegin(), h_data.cend(), cpu_result.begin(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());
  thrust::transform_reduce_into(
    d_data.cbegin(), d_data.cend(), gpu_result.begin(), ::cuda::std::negate<T>(), init, ::cuda::std::plus<T>());

  ASSERT_ALMOST_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestTransformReduceIntoFromConst);

template <class Vector>
void TestTransformReduceIntoCountingIterator()
{
  using T     = typename Vector::value_type;
  using space = typename thrust::iterator_system<typename Vector::iterator>::type;

  thrust::counting_iterator<T, space> first(1);

  Vector result(1);

  thrust::transform_reduce_into(
    first, first + 3, result.begin(), ::cuda::std::negate<short>(), 0, ::cuda::std::plus<short>());

  ASSERT_EQUAL(result[0], -6);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestTransformReduceIntoCountingIterator);
