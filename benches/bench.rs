 #![feature(test)]

extern crate ndarray;
extern crate rand;
extern crate ndarray_rand;
extern crate test;
extern crate rayon;
extern crate simple_parallel;
extern crate ndarray_rblas;
extern crate rblas;
extern crate num;
extern crate rsmat;

use rsmat::dot::dot::*;
use ndarray::{OwnedArray, ArrayView, ArrayViewMut, Ix, ShapeError, arr2};

use ndarray_rand::RandomExt;
use rand::distributions::Range;
use test::Bencher;




pub const M: usize = 128;
pub const K: usize = 100;
pub const N: usize = 128;

#[bench]
fn bench_dot(b: &mut Bencher) {
    let mut c = OwnedArray::zeros((M, N));
    let x = OwnedArray::random((M, K), Range::new(0., 10.));
    let y = OwnedArray::random((K, N), Range::new(0., 10.));
    b.iter(|| matrix_dot_safe(&x.view(), &y.view(), &mut c.view_mut()));
}


#[bench]
fn bench_dot_rayon(b: &mut Bencher) {
    let mut c = OwnedArray::zeros((M, N));
    let x = OwnedArray::random((M, K), Range::new(0., 10.));
    let y = OwnedArray::random((K, N), Range::new(0., 10.));
    b.iter(|| matrix_dot_rayon(&x.view(), &y.view(), &mut c.view_mut()));
}

#[bench]
fn bench_dot_simple_parallel(b: &mut Bencher) {
    let mut c = OwnedArray::zeros((M, N));
    let x = OwnedArray::random((M, K), Range::new(0., 10.));
    let y = OwnedArray::random((K, N), Range::new(0., 10.));
    b.iter(|| matrix_dot_simple_parallel(&x.view(), &y.view(), &mut c.view_mut()));
}
