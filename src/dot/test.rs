#[cfg(test)]

mod tests {
    use dot::dot::*;
    use ndarray::OwnedArray;
    use ndarray_rand::RandomExt;
    use rand::distributions::Range;



    #[test]
    fn test_rayon() {
        let mut c0 = OwnedArray::zeros((10, 20));
        let mut c = OwnedArray::zeros((10, 20));

        let x = OwnedArray::random((10, 5), Range::new(0., 10.));
        let y = OwnedArray::random((5, 20), Range::new(0., 10.));

        matrix_dot_safe(&x.view(), &y.view(), &mut c0.view_mut());
        matrix_dot_rayon(&x.view(), &y.view(), &mut c.view_mut());

        assert!(c == c0);
        assert!(!c.all_close(&OwnedArray::zeros((10, 20)), 0.01))
    }

    #[test]
    fn test_simple_parallel() {
        let mut c0 = OwnedArray::zeros((10, 20));
        let mut c = OwnedArray::zeros((10, 20));

        let x = OwnedArray::random((10, 5), Range::new(0., 10.));
        let y = OwnedArray::random((5, 20), Range::new(0., 10.));

        matrix_dot_safe(&x.view(), &y.view(), &mut c0.view_mut());
        matrix_dot_rayon(&x.view(), &y.view(), &mut c.view_mut());

        assert!(c == c0);
        assert!(!c.all_close(&OwnedArray::zeros((10, 20)), 0.01))
    }



}
