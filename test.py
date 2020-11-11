import os
pwd = os.getcwd()
print(pwd)
todir = os.path.join(pwd, 'aaaaa')
print(todir)
# os.mkdir(todir)
print(os.path.abspath(os.path.join(todir, '../')))
print(os.path.exists(os.path.abspath(os.path.join(todir, '../'))))
print(os.listdir(pwd))