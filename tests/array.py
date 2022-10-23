from pure_array import array, array_equal


# PACKAGE FUNCTIONS
def test_package_array_equal():
    assert array_equal(array([1, 2, 3]), array([1, 2, 3]))
    same = array([1, 2, 3])
    assert array_equal(same, same)
    assert array_equal(array([1, 2, 3], shape=(3, 1)), array([1, 2, 3], shape=(3, 1)))
    assert array_equal(array([1, 2, 3], shape=(1, 3)), array([1, 2, 3], shape=(1, 3)))
    assert array_equal(array([1, 2, 3], shape=(3, 1)), array([[1], [2], [3]]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)), array([[1, 2, 3]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]), array([[1, 2, 3], [4, 5, 6]]))
    assert array_equal(array([1, 2, 3, 4, 5, 6], shape=(2, 3)), array([[1, 2, 3], [4, 5, 6]]))

    assert not array_equal(array([1, 2, 3]), array([1, 2, 4]))
    assert not array_equal(array([1, 2, 3]), array([1, 2, 3, 4]))
    assert not array_equal(array([1, 2, 3], shape=(3, 1)), array([1, 2, 3], shape=(1, 3)))
    assert not array_equal(array([1, 2, 3], shape=(3, 1)), array([[1], [2], [4]]))
    assert not array_equal(array([1, 2, 3], shape=(1, 3)), array([[1, 2, 4]]))
    assert not array_equal(array([[1, 2, 3], [4, 5, 6]]), array([[1, 2, 3], [4, 5, 7]]))
    assert not array_equal(array([1, 2, 3, 4, 5, 6], shape=(2, 3)), array([[1, 2, 3], [4, 5, 7]]))

    assert array_equal(array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
                       array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    assert array_equal(array(list(range(1, 13)), shape=(2, 2, 3)),
                       array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))


def test_package_flatten():
    from pure_array import flatten

    assert flatten([1, 2, 3]) == [1, 2, 3]
    assert flatten([[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
    assert flatten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_package_unravel_index():
    from pure_array import unravel_index

    assert unravel_index(0, (2, 3)) == (0, 0)
    assert unravel_index(5, (2, 3)) == (1, 2)
    assert unravel_index(0, (2, 3, 4)) == (0, 0, 0)
    assert unravel_index(23, (2, 3, 4)) == (1, 2, 3)


def test_package_log():
    from pure_array import log

    assert array_equal(log(array([1, 2, 3])), array([0, 0.6931471805599453, 1.0986122886681098]))
    assert array_equal(log(array([1, 2, 3], shape=(3, 1))), array([[0], [0.6931471805599453], [1.0986122886681098]]))
    assert array_equal(log(array([1, 2, 3], shape=(1, 3))), array([[0, 0.6931471805599453, 1.0986122886681098]]))
    assert array_equal(log(array([[1, 2, 3], [4, 5, 6]])), array(
        [[0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055]]))
    assert array_equal(log(array([1, 2, 3, 4, 5, 6], shape=(2, 3))), array(
        [[0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055]]))


# CLASS FUNCTIONS
def test_get_item():
    assert array_equal(array([1, 2, 3])[0], 1)
    assert array_equal(array([1, 2, 3])[:2], array([1, 2]))
    assert array_equal(array([1, 2, 3])[1:], array([2, 3]))
    assert array_equal(array([1, 2, 3])[:], array([1, 2, 3]))

    assert array_equal(array([[1, 2, 3], [4, 5, 6]])[0], array([1, 2, 3]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]])[0, 0], 1)
    assert array_equal(array([[1, 2, 3], [4, 5, 6]])[:, 0], array([1, 4]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]])[0, :], array([1, 2, 3]))


def test_set_item():
    a = array([1, 2, 3])
    a[0] = 4
    assert array_equal(a, array([4, 2, 3]))
    a[1:] = array([5, 6])
    assert array_equal(a, array([4, 5, 6]))
    a[:] = array([1, 2, 3])
    assert array_equal(a, array([1, 2, 3]))

    a = array([[1, 2, 3], [4, 5, 6]])
    a[0] = array([7, 8, 9])
    assert array_equal(a, array([[7, 8, 9], [4, 5, 6]]))
    a[0, 0] = 10
    assert array_equal(a, array([[10, 8, 9], [4, 5, 6]]))
    a[:, 0] = array([11, 12])
    assert array_equal(a, array([[11, 8, 9], [12, 5, 6]]))
    a[0, :] = array([13, 14, 15])
    assert array_equal(a, array([[13, 14, 15], [12, 5, 6]]))


def test_len():
    assert len(array([1, 2, 3])) == 3
    assert len(array([1, 2, 3], shape=(3, 1))) == 3
    assert len(array([1, 2, 3], shape=(1, 3))) == 3
    assert len(array([[1, 2, 3], [4, 5, 6]])) == 6
    assert len(array([1, 2, 3, 4, 5, 6], shape=(2, 3))) == 6


def test_iter():
    a = array([1, 2, 3])
    for i, x in enumerate(a):
        assert i == x - 1

    a = array([[1, 2, 3], [4, 5, 6]])
    for i, row in enumerate(a):
        for j, x in enumerate(row):
            assert i * 3 + j + 1 == x

    a = array([1, 2, 3, 4, 5, 6], shape=(2, 3))
    for i, row in enumerate(a):
        for j, x in enumerate(row):
            assert i * 3 + j + 1 == x


def test_ndim():
    assert array([1, 2, 3]).ndim() == 1
    assert array([1, 2, 3], shape=(3, 1)).ndim() == 2
    assert array([1, 2, 3], shape=(1, 3)).ndim() == 2
    assert array([[1, 2, 3], [4, 5, 6]]).ndim() == 2
    assert array([1, 2, 3, 4, 5, 6], shape=(2, 3)).ndim() == 2


def test_transpose():
    assert array_equal(array([1, 2, 3]).T, array([1, 2, 3]))
    assert array_equal(array([1, 2, 3], shape=(3, 1)).T, array([1, 2, 3], shape=(1, 3)))
    assert array_equal(array([1, 2, 3], shape=(1, 3)).T, array([1, 2, 3], shape=(3, 1)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).T, array([[1, 4], [2, 5], [3, 6]]))
    assert array_equal(array([1, 2, 3, 4, 5, 6], shape=(2, 3)).T, array([[1, 4], [2, 5], [3, 6]]))


def test_add():
    assert array_equal(array([1, 2, 3]) + array([1, 2, 3]), array([2, 4, 6]))
    assert array_equal(array([1, 2, 3]) + 1, array([2, 3, 4]))
    assert array_equal(1 + array([1, 2, 3]), array([2, 3, 4]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)) + array([1, 2, 3], shape=(1, 3)), array([2, 4, 6], shape=(1, 3)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) + array([[1, 2, 3], [4, 5, 6]]), array([[2, 4, 6], [8, 10, 12]]))

    assert array_equal(array([[1, 2], [3, 4]]) + array([[1], [2]]), array([[2, 3], [5, 6]]))
    assert array_equal(array([[1, 2]]) + array([[1, 2], [3, 4]]), array([[2, 4], [4, 6]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) + array([[1, 2, 3]]), array([[2, 4, 6], [5, 7, 9]]))
    assert array_equal(array([[1, 2, 3]]) + array([[1, 2, 3], [4, 5, 6]]), array([[2, 4, 6], [5, 7, 9]]))


def test_sub():
    assert array_equal(array([2, 4, 6]) - array([1, 2, 3]), array([1, 2, 3]))
    assert array_equal(array([1, 2, 3]) - 1, array([0, 1, 2]))
    assert array_equal(1 - array([1, 2, 3]), array([0, -1, -2]))
    assert array_equal(array([2, 4, 6], shape=(1, 3)) - array([1, 2, 3], shape=(1, 3)), array([1, 2, 3], shape=(1, 3)))
    assert array_equal(array([[2, 4, 6], [8, 10, 12]]) - array([[1, 2, 3], [4, 5, 6]]), array([[1, 2, 3], [4, 5, 6]]))

    assert array_equal(array([[2, 3], [5, 6]]) - array([[1], [2]]), array([[1, 2], [3, 4]]))
    assert array_equal(array([[2, 4], [4, 6]]) - array([[1, 2], [3, 4]]), array([[1, 2], [1, 2]]))
    assert array_equal(array([[2, 4, 6], [5, 7, 9]]) - array([[1, 2, 3]]), array([[1, 2, 3], [4, 5, 6]]))
    assert array_equal(array([[2, 4, 6], [5, 7, 9]]) - array([[1, 2, 3], [4, 5, 6]]), array([[1, 2, 3], [1, 2, 3]]))


def test_mul():
    assert array_equal(array([1, 2, 3]) * array([1, 2, 3]), array([1, 4, 9]))
    assert array_equal(array([1, 2, 3]) * 2, array([2, 4, 6]))
    assert array_equal(2 * array([1, 2, 3]), array([2, 4, 6]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)) * array([1, 2, 3], shape=(1, 3)), array([1, 4, 9], shape=(1, 3)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) * array([[1, 2, 3], [4, 5, 6]]), array([[1, 4, 9], [16, 25, 36]]))

    assert array_equal(array([[1, 2], [3, 4]]) * array([[1], [2]]), array([[1, 2], [6, 8]]))
    assert array_equal(array([[1, 2]]) * array([[1, 2], [3, 4]]), array([[1, 4], [3, 8]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) * array([[1, 2, 3]]), array([[1, 4, 9], [4, 10, 18]]))
    assert array_equal(array([[1, 2, 3]]) * array([[1, 2, 3], [4, 5, 6]]), array([[1, 4, 9], [4, 10, 18]]))


def test_div():
    assert array_equal(array([1, 2, 3]) / array([1, 2, 3]), array([1, 1, 1]))
    assert array_equal(array([1, 2, 3]) / 2, array([0.5, 1, 1.5]))
    assert array_equal(2 / array([1, 2, 3]), array([2, 1, 2 / 3]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)) / array([1, 2, 3], shape=(1, 3)), array([1, 1, 1], shape=(1, 3)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) / array([[1, 2, 3], [4, 5, 6]]), array([[1, 1, 1], [1, 1, 1]]))

    assert array_equal(array([[1, 2], [3, 4]]) / array([[1], [2]]), array([[1, 2], [1.5, 2]]))
    assert array_equal(array([[1, 2]]) / array([[1, 2], [3, 4]]), array([[1, 1], [1 / 3, 1 / 2]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) / array([[1, 2, 3]]), array([[1, 1, 1], [4, 2.5, 2]]))
    assert array_equal(array([[1, 2, 3]]) / array([[1, 2, 3], [4, 5, 6]]), array([[1, 1, 1], [1 / 4, 2 / 5, 1 / 2]]))


def test_floor_div():
    assert array_equal(array([1, 2, 3]) // array([1, 2, 3]), array([1, 1, 1]))
    assert array_equal(array([1, 2, 3]) // 2, array([0, 1, 1]))
    assert array_equal(2 // array([1, 2, 3]), array([2, 1, 0]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)) // array([1, 2, 3], shape=(1, 3)), array([1, 1, 1], shape=(1, 3)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) // array([[1, 2, 3], [4, 5, 6]]), array([[1, 1, 1], [1, 1, 1]]))

    assert array_equal(array([[1, 2], [3, 4]]) // array([[1], [2]]), array([[1, 2], [1, 2]]))
    assert array_equal(array([[1, 2]]) // array([[1, 2], [3, 4]]), array([[1, 1], [0, 0]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) // array([[1, 2, 3]]), array([[1, 1, 1], [4, 2, 2]]))
    assert array_equal(array([[1, 2, 3]]) // array([[1, 2, 3], [4, 5, 6]]), array([[1, 1, 1], [0, 0, 0]]))


def test_mod():
    assert array_equal(array([1, 2, 3]) % array([1, 2, 3]), array([0, 0, 0]))
    assert array_equal(array([1, 2, 3]) % 2, array([1, 0, 1]))
    assert array_equal(2 % array([1, 2, 3]), array([0, 0, 2]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)) % array([1, 2, 3], shape=(1, 3)), array([0, 0, 0], shape=(1, 3)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) % array([[1, 2, 3], [4, 5, 6]]), array([[0, 0, 0], [0, 0, 0]]))

    assert array_equal(array([[1, 2], [3, 4]]) % array([[1], [2]]), array([[0, 0], [1, 0]]))
    assert array_equal(array([[1, 2]]) % array([[1, 2], [3, 4]]), array([[0, 0], [1, 2]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) % array([[1, 2, 3]]), array([[0, 0, 0], [0, 1, 0]]))
    assert array_equal(array([[1, 2, 3]]) % array([[1, 2, 3], [4, 5, 6]]), array([[0, 0, 0], [1, 2, 3]]))


def test_pow():
    assert array_equal(array([1, 2, 3]) ** array([1, 2, 3]), array([1, 4, 27]))
    assert array_equal(array([1, 2, 3]) ** 2, array([1, 4, 9]))
    assert array_equal(2 ** array([1, 2, 3]), array([2, 4, 8]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)) ** array([1, 2, 3], shape=(1, 3)),
                       array([1, 4, 27], shape=(1, 3)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) ** array([[1, 2, 3], [4, 5, 6]]),
                       array([[1, 4, 27], [256, 3125, 46656]]))

    assert array_equal(array([[1, 2], [3, 4]]) ** array([[1], [2]]), array([[1, 2], [9, 16]]))
    assert array_equal(array([[1, 2]]) ** array([[1, 2], [3, 4]]), array([[1, 4], [1, 16]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) ** array([[1, 2, 3]]), array([[1, 4, 27], [4, 25, 216]]))
    assert array_equal(array([[1, 2, 3]]) ** array([[1, 2, 3], [4, 5, 6]]), array([[1, 4, 27], [1, 32, 729]]))


def test_neg():
    assert array_equal(-array([1, 2, 3]), array([-1, -2, -3]))
    assert array_equal(-array([1, 2, 3], shape=(1, 3)), array([-1, -2, -3], shape=(1, 3)))
    assert array_equal(-array([[1, 2, 3], [4, 5, 6]]), array([[-1, -2, -3], [-4, -5, -6]]))


def test_abs():
    assert array_equal(abs(array([1, -2, 3])), array([1, 2, 3]))
    assert array_equal(abs(array([1, -2, 3], shape=(1, 3))), array([1, 2, 3], shape=(1, 3)))
    assert array_equal(abs(array([[1, -2, 3], [-4, 5, -6]])), array([[1, 2, 3], [4, 5, 6]]))


def test_matmul():
    assert array_equal(array([1, 2, 3]) @ array([1, 2, 3]), array([14]))
    assert array_equal(array([1, 2, 3], shape=(3, 1)) @ array([1, 2, 3], shape=(1, 3)),
                       array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) @ array([[1, 2, 3], [4, 5, 6]]).T, array([[14, 32], [32, 77]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).T @ array([[1, 2, 3], [4, 5, 6]]),
                       array([[17, 22, 27], [22, 29, 36], [27, 36, 45]]))


def test_tolist():
    assert array([1, 2, 3]).tolist() == [1, 2, 3]
    assert array([1, 2, 3], shape=(1, 3)).tolist() == [[1, 2, 3]]
    assert array([[1, 2, 3], [4, 5, 6]]).tolist() == [[1, 2, 3], [4, 5, 6]]
    assert array([[1, 2, 3], [4, 5, 6]]).T.tolist() == [[1, 4], [2, 5], [3, 6]]


def test_reshape():
    assert array_equal(array([1, 2, 3]).reshape(3, 1), array([1, 2, 3], shape=(3, 1)))
    assert array_equal(array([1, 2, 3], shape=(1, 3)).reshape(3, 1), array([1, 2, 3], shape=(3, 1)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2), array([[1, 2], [3, 4], [5, 6]], shape=(3, 2)))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2).reshape(2, 3),
                       array([[1, 2, 3], [4, 5, 6]], shape=(2, 3)))


def test_flatten():
    assert array_equal(array([1, 2, 3]).flatten(), array([1, 2, 3]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)).flatten(), array([1, 2, 3]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).flatten(), array([1, 2, 3, 4, 5, 6]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).T.flatten(), array([1, 4, 2, 5, 3, 6]))


def test_sum():
    assert array([1, 2, 3]).sum() == 6
    assert array([1, 2, 3], shape=(1, 3)).sum() == 6
    assert array([[1, 2, 3], [4, 5, 6]]).sum() == 21
    assert array([[1, 2, 3], [4, 5, 6]]).T.sum() == 21

    assert array([1, 2, 3]).sum(axis=0) == 6
    assert array_equal(array([1, 2, 3], shape=(1, 3)).sum(axis=0), array([1, 2, 3]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).sum(axis=0), array([5, 7, 9]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).sum(axis=1), array([6, 15]))

    # 3D
    assert array_equal(array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).sum(axis=0),
                       array([[8, 10, 12], [14, 16, 18]]))
    assert array_equal(array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).sum(axis=1),
                       array([[5, 7, 9], [17, 19, 21]]))
    assert array_equal(array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).sum(axis=2),
                       array([[6, 15], [24, 33]]))


def test_max():
    assert array([1, 2, 3]).max() == 3
    assert array([1, 2, 3], shape=(1, 3)).max() == 3
    assert array([[1, 2, 3], [4, 5, 6]]).max() == 6
    assert array([[1, 2, 3], [4, 5, 6]]).T.max() == 6

    assert array([1, 2, 3]).max(axis=0) == 3
    assert array_equal(array([1, 2, 3], shape=(1, 3)).max(axis=0), array([1, 2, 3]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).max(axis=0), array([4, 5, 6]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).max(axis=1), array([3, 6]))


def test_argmax():
    assert array([1, 2, 3]).argmax() == (2,)
    assert array([1, 2, 3], shape=(1, 3)).argmax() == (0, 2)
    assert array([[1, 2, 3], [4, 5, 6]]).argmax() == (1, 2)
    assert array([[1, 2, 3], [4, 5, 6]]).T.argmax() == (2, 1)

    assert array([1, 2, 3]).argmax(axis=0) == (2,)
    assert array([1, 2, 3], shape=(1, 3)).argmax(axis=0) == (0, 0, 0)
    assert array([[1, 2, 7], [4, 5, 6]]).argmax(axis=0) == (1, 1, 0)
    assert array([[1, 2, 3], [4, 5, 6]]).argmax(axis=1) == (2, 2)


def test_append():
    assert array_equal(array([1, 2, 3]).append(array([4, 5, 6])), array([1, 2, 3, 4, 5, 6]))
    assert array_equal(array([1, 2, 3], shape=(1, 3)).append(array([4, 5, 6])), array([[1, 2, 3], [4, 5, 6]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).append(array([7, 8, 9])), array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]).append(array([7, 8, 9], shape=(1, 3))),
                       array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # TODO: catch error
    # assert not array_equal(array([[1, 2, 3], [4, 5, 6]]).append(array([7, 8, 9], shape=(3, 1))),
    #                        array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    a = array([[1, 2, 3], [4, 5, 6]])
    assert array_equal(a.append(array([7, 8, 9])), array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert array_equal(a, array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


# TODO: MORE TESTS for boolean operations and indexing
def test_eq():
    assert array_equal(array([1, 2, 3]) == array([1, 2, 3]), array([True, True, True]))
    assert array_equal(array([1, 2, 3]) == array([1, 2, 4]), array([True, True, False]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) == array([[1, 2, 3], [4, 5, 6]]), array([[True, True, True], [True, True, True]]))
    assert array_equal(array([[1, 2, 3], [4, 5, 6]]) == array([[1, 2, 3]]), array([[True, True, True], [False, False, False]]))
