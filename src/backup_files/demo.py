



def func(fargs1,fargs2, *args, **kwargs):  # fargs: formal args, kargs: keywords arguments

    print(fargs1, fargs2, args, kwargs)
    if kwargs is not None:
        for key, value in kwargs.items():
            print("%s == %s" %(key,value))


if __name__ == '__main__':

    a=0
    func(a,'2','1',a, c=3, b=5)