
extern crate ndarray;
extern crate rand;
extern crate ndarray_rand;

use ndarray::{Axis, ArrayView,ArrayViewMut, Ix};
use rayon;
use rayon::prelude::*;
use simple_parallel;

// various array views for divide-and-conquering
pub type VectorView<'a,A> = ArrayView<'a,A, Ix>;
pub type MatView<'a,A> = ArrayView<'a,A,(Ix,Ix)>;
pub type MatViewMut<'a,A> = ArrayViewMut<'a,A,(Ix,Ix)>;


// basic vector dot product
pub fn vector_dot(left : VectorView<f64>,right:  VectorView<f64>) -> f64 {
    unsafe{
        (0..right.len()).fold(0.0, |x, y| x + *right.uget(y) * *left.uget(y) )
    }

}

// basic matrix multiplication
pub fn matrix_dot( left : &MatView<f64>, right: &MatView<f64>, init : &mut MatViewMut<f64>){
    let (m,k1)= left.dim();
    let (k2,n) = right.dim();
    assert_eq!(k1, k2);
    for ix in 0..m{
        for jx in 0..n{
            let left_row = left.row(ix);
            let right_col = right.column(jx);
            unsafe{
                let mut value = init.uget_mut((ix,jx));
                *value += vector_dot(left_row, right_col);
            }
        }
    }
}



pub const BLOCKSIZE : usize = 50;

// parallelized matrix multiplication via rayon.
pub fn matrix_dot_rayon(left: &MatView<f64>, right : &MatView<f64>,  init : &mut MatViewMut<f64>){

    let (m, k1) = left.dim();
    let (k2, n) = right.dim();
    assert_eq!(k1, k2);

    if m <= BLOCKSIZE && n <= BLOCKSIZE {
        matrix_dot(left,right,init);
        return;
    } else{
        if m > BLOCKSIZE {
           let mid = m / 2;
           let (left_0, left_1) = left.split_at(Axis(0), mid);
           let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(0), mid);
           rayon::join(|| matrix_dot_rayon(&left_0, right, &mut init_left),
                       || matrix_dot_rayon(&left_1, right, &mut init_right));

       } else if n > BLOCKSIZE {
           let mid = n / 2;
           let (right_0, right_1) = right.split_at(Axis(1), mid);
           let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(1), mid);
           rayon::join(|| matrix_dot_rayon(left,&right_0, &mut init_left),
                       || matrix_dot_rayon(left, &right_1, &mut init_right));
        }
    }
}


// parallelized matrix multiplication via simple_parallel.
pub fn matrix_dot_simple_parallel(left: &MatView<f64>, right : &MatView<f64>,  init : &mut MatViewMut<f64>){

    let (m, k1) = left.dim();
    let (k2, n) = right.dim();


    assert_eq!(k1, k2);
    if m <= BLOCKSIZE && n <= BLOCKSIZE {
        matrix_dot(left,right,init);
        return;
    } else{
        if m > BLOCKSIZE {
           let mid = m / 2;
           let (left_0, left_1) = left.split_at(Axis(0), mid);
           let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(0), mid);
           simple_parallel::both((&left_0, right, &mut init_left),
                                (&left_1, right, &mut init_right),
                                |(x,y,z)| matrix_dot_simple_parallel(x,y,z));

       } else if n > BLOCKSIZE {
           let mid = n / 2;
           let (right_0, right_1) = right.split_at(Axis(1), mid);
           let (mut init_left, mut init_right) = init.view_mut().split_at(Axis(1), mid);
           simple_parallel::both((left, &right_0, &mut init_left),
                                (left, &right_1, &mut init_right),
                                |(x,y,z)| matrix_dot_simple_parallel(x,y,z));

        }
    }
}
