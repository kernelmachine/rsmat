#[cfg(test)]

mod tests {
    use util::util::*;
    use ndarray::{OwnedArray, arr2};
    use ndarray_rand::RandomExt;
    use rand::distributions::Range;
    use rand::{thread_rng, Rng};
    use rayon::prelude::*;
    use test::Bencher;


    #[test]
    fn test_par_dot(){
        let mut c0 = OwnedArray::zeros((100,50));

        let mut c = OwnedArray::zeros((100,50));
        let mut x = OwnedArray::random((100,20), Range::new(0.,10.));
        let mut y = OwnedArray::random((20,50), Range::new(0.,10.));

        matrix_dot(&x.view(), &y.view(),&mut c0.view_mut());
        matrix_dot_par(&x.view(), &y.view(),&mut c.view_mut());

        assert!(c == c0);
        assert!(c.all_close(&OwnedArray::zeros((100,50)), 0.01) == false)
    }

    #[bench]
    fn bench_par_dot(b: &mut Bencher) {
        let mut c = OwnedArray::zeros((100,100));
        let mut x = OwnedArray::random((100,100), Range::new(0.,10.));
        let mut y = OwnedArray::random((100,100), Range::new(0.,10.));
        b.iter(|| {
            matrix_dot_par(&x.view(), &y.view(),&mut c.view_mut())
        });
    }

    #[bench]
    fn bench_dot(b: &mut Bencher) {
        let mut c = OwnedArray::zeros((100,100));
        let mut x = OwnedArray::random((100,100), Range::new(0.,10.));
        let mut y = OwnedArray::random((100,100), Range::new(0.,10.));
        b.iter(|| {
            matrix_dot(&x.view(), &y.view(),&mut c.view_mut())
        });
    }

    // #[bench]
    // fn bench_dot(b: &mut Bencher) {
    //
    //     let mut x = OwnedArray::random((100,100), Range::new(0.,10.));
    //     let mut y = OwnedArray::random((100,100), Range::new(0.,10.));
    //     b.iter(|| {
    //         matrix_dot(&mut x, &mut y)
    //     });
    // }


}
