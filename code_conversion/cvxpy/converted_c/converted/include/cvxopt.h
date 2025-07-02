
/*
 * Copyright 2012-2023 M. Andersen and L. Vandenberghe.
 * Copyright 2010-2011 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT.
 *
 * CVXOPT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * CVXOPT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "blas_redefines.h"
#include "assert.h"
#include "stddef.h"
#include "stdbool.h"

/* ANSI99 complex is disabled during build of CHOLMOD */

#if !defined(NO_ANSI99_COMPLEX) 
#include "complex.h"
#if !defined(_MSC_VER)
#define MAT_BUFZ(O)  ((double complex *)((matrix *)O)->buffer)
#else
#define MAT_BUFZ(O)  ((_Dcomplex *)((matrix *)O)->buffer)
#endif
#endif

#ifndef __CVXOPT__
#define __CVXOPT__

#define INT           0
#define DOUBLE        1
#define COMPLEX       2
#define MAT_DENSE     3
#define MAT_SPARSE    4

#define int_t  ptrdiff_t

typedef struct {
  int  mat_type;        // MAT_DENSE
  void *buffer;         /* in column-major-mode array of type 'id' */
  int  nrows, ncols;    /* number of rows and columns */
  int  id;              /* DOUBLE, INT, COMPLEX */
} matrix;

typedef struct {
  void  *values;        /* value list */
  int_t *colptr;        /* column pointer list */
  int_t *rowind;        /* row index list */
  int_t nrows, ncols;   /* number of rows and columns */
  int   id;             /* DOUBLE, COMPLEX */
} ccs;

typedef struct {
  int mat_type;         // MAT_SPARSE
  ccs *obj;
} spmatrix;

/* Function prototypes for matrix operations */
matrix *Matrix_New(int, int, int);
matrix *Matrix_NewFromMatrix(matrix *, int);
matrix *Matrix_NewFromList(void *, int);

spmatrix *SpMatrix_New(int_t, int_t, int_t, int);
spmatrix *SpMatrix_NewFromSpMatrix(spmatrix *, int);
spmatrix *SpMatrix_NewFromIJV(matrix *, matrix *, matrix *, int_t, int_t, int);


static inline int is_matrix(void *ptr) {
    if (!ptr) return 0;
    return ((matrix *)ptr)->mat_type == MAT_DENSE;
}

static inline int is_spmatrix(void *ptr) {
    if (!ptr) return 0;
    return ((spmatrix *)ptr)->mat_type == MAT_SPARSE;
}

#define Matrix_Check(MAT) is_matrix(MAT)
#define SpMatrix_Check(MAT) is_spmatrix(MAT)

/*
 * Below this line are non-essential convenience macros
 */

#define MAT_BUF(O)   ((matrix *)O)->buffer
#define MAT_BUFI(O)  ((int_t *)((matrix *)O)->buffer)
#define MAT_BUFD(O)  ((double *)((matrix *)O)->buffer)
#ifndef _MSC_VER
#define MAT_BUFZ(O)  ((double complex *)((matrix *)O)->buffer)
#else
#define MAT_BUFZ(O)  ((_Dcomplex *)((matrix *)O)->buffer)
#endif
#define MAT_NROWS(O) ((matrix *)O)->nrows
#define MAT_NCOLS(O) ((matrix *)O)->ncols
#define MAT_LGT(O)   (MAT_NROWS(O)*MAT_NCOLS(O))
#define MAT_ID(O)    ((matrix *)O)->id

#define SP_NCOLS(O)  ((spmatrix *)O)->obj->ncols
#define SP_NROWS(O)  ((spmatrix *)O)->obj->nrows
#define SP_LGT(O)    (SP_NROWS(O)*SP_NCOLS(O))
#define SP_NNZ(O)    ((spmatrix *)O)->obj->colptr[SP_NCOLS(O)]
#define SP_ID(O)     ((spmatrix *)O)->obj->id
#define SP_COL(O)    ((spmatrix *)O)->obj->colptr
#define SP_ROW(O)    ((spmatrix *)O)->obj->rowind
#define SP_VAL(O)    ((spmatrix *)O)->obj->values
#define SP_VALD(O)   ((double *)((spmatrix *)O)->obj->values)
#ifndef _MSC_VER
#define SP_VALZ(O)   ((double complex *)((spmatrix *)O)->obj->values)
#else
#define SP_VALZ(O)   ((_Dcomplex *)((spmatrix *)O)->obj->values)
#endif

#define CCS_NROWS(O) ((ccs *)O)->nrows
#define CCS_NCOLS(O) ((ccs *)O)->ncols
#define CCS_NNZ(O)   ((ccs *)O)->colptr[CCS_NCOLS(O)]

#endif

