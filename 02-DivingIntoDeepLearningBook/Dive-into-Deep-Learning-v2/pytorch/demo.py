def fun_00(i, str_input):
    print(i, str_input)


def fun_01(str_input):
    for i in range(3):
        fun_00(i, str_input)


# if __name__ == '__main__':

print('Start...')
fun_01('abc')
print('Start...')
