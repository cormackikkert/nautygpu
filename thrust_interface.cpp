#include <thrust/sort.h>

extern "C" {
    void thrust_stable_sort_by_key(int* key, int* arr, int N) {
        thrust::stable_sort_by_key(key, key+N, arr);
    }
}
