
extern crate ndarray;
extern crate rand;
extern crate ndarray_rand;

use ndarray::{OwnedArray,Axis, ArrayView,ArrayViewMut, Ix};
use rayon;
use rayon::prelude::*;


/// Rectangular matrix.
pub type Vector<'a,A> = ArrayView<'a,A, Ix>;
pub type Mat<A> = OwnedArray<A, (Ix,Ix)>;

pub fn vector_dot(lhs : Vector<f64>,rhs: Vector<f64>) -> f64 {
    debug_assert_eq!(rhs.len(), lhs.len());
    (0..rhs.len()).fold(0.0, |x, y| x + *rhs.get(y).expect("lhs index error") * *lhs.get(y).expect("lhs index error") )
}

pub fn matrix_dot( lhs : &ArrayView<f64,(Ix,Ix)>, rhs: &ArrayView<f64,(Ix,Ix)>,df : &mut ArrayViewMut<f64,(Ix,Ix)>){
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

pub fn matrix_dot_par(lhs: &ArrayView<f64,(Ix,Ix)>, rhs : &ArrayView<f64,(Ix,Ix)>,  df : &mut ArrayViewMut<f64,(Ix,Ix)>){

    let ((m, k), (k2, n)) = (lhs.dim(), rhs.dim());


    debug_assert_eq!(k, k2);
    if m < 5 && n < 5 {

        matrix_dot(lhs,rhs,df);
        return;
    }
    else{
        if m > 5 {
           let mid = m / 2;
           let (a0, a1) = lhs.view().split_at(Axis(0), mid);
           let (mut df0, mut df1) = df.view_mut().split_at(Axis(0), mid);

           rayon::join(|| matrix_dot_par(&a0, rhs, &mut df0),
                       || matrix_dot_par(&a1, rhs, &mut df1));
       } else if n > 5 {
           let mid = n / 2;
           let (b0, b1) = rhs.view().split_at(Axis(1), mid);
           let (mut df0, mut df1) = df.view_mut().split_at(Axis(1), mid);
           rayon::join(|| matrix_dot_par(lhs,&b0, &mut df0),
                       || matrix_dot_par(lhs, &b1, &mut df1));
    }
    }
}
//
//
// pub fn dot(rhs: Mat<f64>, lhs : Mat<f64>) {
//     mid = rhs
//     let (a0, a1) = lhs.subview(Axis(0), 0).split_at(Axis(0), 1);
//     let (a0, a1) = lhs.split_at(Axis(0), mid);
//     let (mut c0, mut c1) = c.view_mut().split_at(Axis(0), mid);
//     rayon::join(move || mat_mul_general(alpha, &a0, rhs, beta, &mut c0),
//                 move || mat_mul_general(alpha, &a1, rhs, beta, &mut c1));
// }
