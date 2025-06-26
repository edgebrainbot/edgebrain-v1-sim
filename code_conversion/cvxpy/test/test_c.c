#include <stdio.h>
#include <stddef.h>
#include <Python.h>

int main() {
    int arr[] = {1,2,3,4,5};
    int *p1 = &arr[1];
    int *p2 = &arr[4];

    ptrdiff_t diff = p2 - p1;
    printf("Pointer difference: %td elements\n", diff);
    printf("sizeof(ptrdiff_t) = %zu\n", sizeof(ptrdiff_t));
    printf("sizeof(Py_ssize_t) = %zu\n", sizeof(Py_ssize_t));
    return 0;
}