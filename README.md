ED(誤差拡散法)のChisel実装、MNISTデータの学習
=======================

[金子勇さんのED法を実装してMNISTを学習させてみた](https://qiita.com/pocokhc/items/f7ab56051bb936740b8f)
を元に移植したもの
MNISTデータの特定の数とそれ以外の判別をサポートしている。

## How to run
```sh
sbt test
```

Alternatively, if you use Mill:
```sh
./mill %NAME%.test
```

You should see a whole bunch of output that ends with something like the following lines
```
[info] Tests: succeeded 1, failed 0, canceled 0, ignored 0, pending 0
[info] All tests passed.
[success] Total time: 5 s, completed Dec 16, 2020 12:18:44 PM
```
If you see the above then...

## 既知の問題点

## Reference
- [金子勇さんのED法を実装してMNISTを学習させてみた](https://qiita.com/pocokhc/items/f7ab56051bb936740b8f)
- [ED法を高速化してその性能をMNISTで検証してみた](https://qiita.com/pocokhc/items/f4387c099a28a69df918)
- [『Winny』の金子勇さんの失われたED法を求めて...いたら見つかりました](https://qiita.com/kanekanekaneko/items/901ee2837401750dfdad)