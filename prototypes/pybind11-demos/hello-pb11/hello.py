from ctypes import c_int, addressof
import hello

def test_add():
    x = 1
    y = 3
    c_x = c_int(x)
    c_y = c_int(y)
    x_id = hex(addressof(c_x))
    y_id = hex(addressof(c_y))
    print("x_id: ", x_id)
    print("y_id: ", y_id)
    x = hello.add(x, y)
    print("x = ", x)
    assert(x == 4)


if __name__ == '__main__':
    test_add()