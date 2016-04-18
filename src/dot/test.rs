#[cfg(test)]

mod tests {
    use dot::dot::*;
    use ndarray::OwnedArray;
    use ndarray_rand::RandomExt;
    use rand::distributions::Range;
    use test::Bencher;



    #[test]
    fn test_rayon(){
        let mut c0 = OwnedArray::zeros((10,20));
        let mut c = OwnedArray::zeros((10,20));

        let x = OwnedArray::random((10,5), Range::new(0.,10.));
        let y = OwnedArray::random((5,20), Range::new(0.,10.));

        matrix_dot(&x.view(), &y.view(),&mut c0.view_mut());
        matrix_dot_rayon(&x.view(), &y.view(),&mut c.view_mut());

        assert!(c == c0);
        assert!(c.all_close(&OwnedArray::zeros((10,20)), 0.01) == false)
    }

    #[test]
    fn test_simple_parallel(){
        let mut c0 = OwnedArray::zeros((10,20));
        let mut c = OwnedArray::zeros((10,20));

        let x = OwnedArray::random((10,5), Range::new(0.,10.));
        let y = OwnedArray::random((5,20), Range::new(0.,10.));

        matrix_dot(&x.view(), &y.view(),&mut c0.view_mut());
        matrix_dot_rayon(&x.view(), &y.view(),&mut c.view_mut());

        assert!(c == c0);
        assert!(c.all_close(&OwnedArray::zeros((10,20)), 0.01) == false)
    }

    pub const INNER_SIZE : usize = 100;
    pub const OUTER_SIZE : usize = 100;

    #[bench]
    fn bench_dot(b: &mut Bencher) {
        let mut c = OwnedArray::zeros((OUTER_SIZE,OUTER_SIZE));
        let x = OwnedArray::random((OUTER_SIZE,INNER_SIZE), Range::new(0.,10.));
        let y = OwnedArray::random((INNER_SIZE,OUTER_SIZE), Range::new(0.,10.));
        b.iter(|| {
            matrix_dot(&x.view(), &y.view(),&mut c.view_mut())
        });
    }


    #[bench]
    fn bench_dot_rayon(b: &mut Bencher) {
        let mut c = OwnedArray::zeros((OUTER_SIZE,OUTER_SIZE));
        let x = OwnedArray::random((OUTER_SIZE,INNER_SIZE), Range::new(0.,10.));
        let y = OwnedArray::random((INNER_SIZE,OUTER_SIZE), Range::new(0.,10.));
        b.iter(|| {
            matrix_dot_rayon(&x.view(), &y.view(),&mut c.view_mut())
        });
    }

    #[bench]
    fn bench_dot_simple_parallel(b: &mut Bencher) {
        let mut c = OwnedArray::zeros((OUTER_SIZE,OUTER_SIZE));
        let x = OwnedArray::random((OUTER_SIZE,INNER_SIZE), Range::new(0.,10.));
        let y = OwnedArray::random((INNER_SIZE,OUTER_SIZE), Range::new(0.,10.));
        b.iter(|| {
            matrix_dot_simple_parallel(&x.view(), &y.view(),&mut c.view_mut())
        });
    }


}
