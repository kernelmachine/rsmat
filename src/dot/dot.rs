
extern crate ndarray;
extern crate rand;
extern crate ndarray_rand;

use ndarray::{Axis, ArrayView,ArrayViewMut, Ix};
use rayon;
use rayon::prelude::*;
use simple_parallel;

pub type VectorView<'a,A> = ArrayView<'a,A, Ix>;
pub type MatView<'a,A> = ArrayView<'a,A,(Ix,Ix)>;
pub type MatViewMut<'a,A> = ArrayViewMut<'a,A,(Ix,Ix)>;

pub const THRESHOLD : usize = 50;

pub fn vector_dot(lhs : VectorView<f64>,rhs: VectorView<f64>) -> f64 {
    debug_assert_eq!(rhs.len(), lhs.len());
    (0..rhs.len()).fold(0.0, |x, y| x + *rhs.get(y).expect("lhs index error") * *lhs.get(y).expect("lhs index error") )
}



pub fn matrix_dot( lhs : &MatView<f64>, rhs: &MatView<f64>,df : &mut MatViewMut<f64>){
    let ((m,k1),(k2,n)) = (lhs.dim(),rhs.dim());

    debug_assert_eq!(k1, k2);
    for ix in 0..m{
        for jx in 0..n{
            let lhs_row = lhs.row(ix);
            let rhs_col = rhs.column(jx);
            let mut value = df.get_mut((ix,jx)).expect("index error");
            *value += vector_dot( lhs_row, rhs_col);
        }
    }
}



pub fn matrix_dot_rayon(lhs: &MatView<f64>, rhs : &MatView<f64>,  df : &mut MatViewMut<f64>){

    let ((m, k), (k2, n)) = (lhs.dim(), rhs.dim());
    debug_assert_eq!(k, k2);
    if m <= THRESHOLD && n <= THRESHOLD {

        matrix_dot(lhs,rhs,df);
        return;
    }
    else{
        if m > THRESHOLD {
           let mid = m / 2;
           let (a0, a1) = lhs.view().split_at(Axis(0), mid);
           let (mut df0, mut df1) = df.view_mut().split_at(Axis(0), mid);
           rayon::join(|| matrix_dot_rayon(&a0, rhs, &mut df0),
                       || matrix_dot_rayon(&a1, rhs, &mut df1));

       } else if n > THRESHOLD {
           let mid = n / 2;
           let (b0, b1) = rhs.view().split_at(Axis(1), mid);
           let (mut df0, mut df1) = df.view_mut().split_at(Axis(1), mid);
           rayon::join(|| matrix_dot_rayon(lhs,&b0, &mut df0),
                       || matrix_dot_rayon(lhs, &b1, &mut df1));
    }
    }
}



pub fn matrix_dot_simple_parallel(lhs: &MatView<f64>, rhs : &MatView<f64>,  df : &mut MatViewMut<f64>){

    let ((m, k), (k2, n)) = (lhs.dim(), rhs.dim());


    debug_assert_eq!(k, k2);
    if m <= THRESHOLD && n <= THRESHOLD {

        matrix_dot(lhs,rhs,df);
        return;
    }
    else{
        if m > THRESHOLD {
           let mid = m / 2;
           let (a0, a1) = lhs.view().split_at(Axis(0), mid);
           let (mut df0, mut df1) = df.view_mut().split_at(Axis(0), mid);
           simple_parallel::both((&a0, rhs, &mut df0),(&a1, rhs, &mut df1), |(x,y,z)| matrix_dot_simple_parallel(x,y,z));

       } else if n > THRESHOLD {
           let mid = n / 2;
           let (b0, b1) = rhs.view().split_at(Axis(1), mid);
           let (mut df0, mut df1) = df.view_mut().split_at(Axis(1), mid);
           simple_parallel::both((lhs, &b0, &mut df0),(lhs, &b1, &mut df1), |(x,y,z)| matrix_dot_simple_parallel(x,y,z));

    }
    }
}
