#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace chrono;

// Sequential Bubble Sort
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; ++i)
        for (int j = 0; j < n-i-1; ++j)
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
}

// Parallel Bubble Sort using OpenMP
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Merge function for Merge Sort
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m-l+1, n2 = r-m;
    vector<int> L(n1), R(n2);
    for(int i = 0; i < n1; i++) L[i] = arr[l+i];
    for(int i = 0; i < n2; i++) R[i] = arr[m+1+i];

    int i = 0, j = 0, k = l;
    while(i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while(i < n1) arr[k++] = L[i++];
    while(j < n2) arr[k++] = R[j++];
}

// Sequential Merge Sort
void mergeSortSequential(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l+r)/2;
        mergeSortSequential(arr, l, m);
        mergeSortSequential(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP
void mergeSortParallel(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr, l, m);

            #pragma omp section
            mergeSortParallel(arr, m+1, r);
        }
        merge(arr, l, m, r);
    }
}

// Function to generate a random array
vector<int> generateRandomArray(int n) {
    vector<int> arr(n);
    for (int& x : arr)
        x = rand() % 10000;
    return arr;
}

// Measure and print performance
template<typename Func>
double measure(Func f, vector<int>& arr) {
    auto start = high_resolution_clock::now();
    f(arr);
    auto stop = high_resolution_clock::now();
    return duration<double>(stop - start).count();
}

void printArray(const vector<int>& arr) {
    cout << "Array Elements:\n";
    for (int i = 0; i < arr.size(); ++i) {
        cout << arr[i] << " ";
    }
}

int main() {
    const int n = 100000; // Try with 10000 for MergeSort comparison
    vector<int> arr1 = generateRandomArray(n);
    vector<int> arr2 = arr1;
    vector<int> arr3 = arr1;
    vector<int> arr4 = arr1;
    // printArray(arr1);

    cout << "\n\nArray Size: " << n << "\n";

    double t1 = measure(bubbleSortSequential, arr1);
    cout << "Sequential Bubble Sort: " << t1 << " s\n";

    double t2 = measure(bubbleSortParallel, arr2);
    cout << "Parallel Bubble Sort:   " << t2 << " s\n";

    double t3 = measure([&](vector<int>& arr){ mergeSortSequential(arr, 0, arr.size()-1); }, arr3);
    cout << "Sequential Merge Sort:  " << t3 << " s\n";

    double t4 = measure([&](vector<int>& arr){ mergeSortParallel(arr, 0, arr.size()-1); }, arr4);
    cout << "Parallel Merge Sort:    " << t4 << " s\n";

    return 0;
}
