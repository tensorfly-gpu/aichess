import sys
import time

import ZODB, ZODB.FileStorage,ZODB.FileStorage.interfaces
import persistent,persistent.list
import transaction
from BTrees.IOBTree import IOBTree
from BTrees.OOBTree import OOBTree
import numpy as np
import zip_array

from config import CONFIG


class MyZODB():
    def __init__(self,data_path='data/train_data_buffer.fs'):
        self.storage = ZODB.FileStorage.FileStorage(data_path,pack_keep_old=False)
        self.db = ZODB.DB(self.storage,cache_size=1,cache_size_bytes=1,large_record_size=1<<32)
        self.connection = self.db.open()
        self.root = self.connection.root()
        while True:
            try:
                if not isinstance(self.root.data,IOBTree):
                    self.root.data = IOBTree()
                    transaction.commit()
                else:
                    break
            except:
                time.sleep(5)
        self.data = self.root.data
    def close(self):
        self.connection.close()
        self.db.close()
        self.storage.close()

    def getMaxIters(self):
        return self.data.maxKey() if len(self.data)>0 else 0
    def getMinIters(self):
        return self.data.minKey() if len(self.data)>0 else 0

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def gc(self,buffer_size = CONFIG['buffer_size']):
        l = [d for d in self.data.values()]
        length = len(l)
        if length > buffer_size:
            print("gc....")
            self.delAny((int)(length / 10))       #删除10%数据
            self.pack()

    def delAny(self,num):
        for i in range(num):
            min_index = self.getMinIters()
            if min_index != 0:
                self.data.__delitem__(min_index)
        transaction.commit()

    def pack(self):
        self.db.pack(time.time())
        transaction.commit()

    #加载所有数据
    def load(self):
        return self.data.maxKey(),list([d for ds in self.data.values() for d in ds])

    def dump(self,iters, data_buffer):
        start = time.time()
        i = iters
        while True:
            if not self.data.has_key(i):
                self.data.insert(i,data_buffer)
                transaction.commit()
                break
            else:
                i += 1
        print("写入花费：",time.time() - start)
        return i

class Book(persistent.Persistent):

   def __init__(self, title):
       self.title = title
       self.authors = []

   def add_author(self, author):
       self.authors.append(author)
       self._p_changed = True

if __name__ == '__main__':
    t = MyZODB('data/train_data_buffer.fs')
    # l = t.root.tmp
    # print(l[0][0][:8])
    # t.dump(1,l)
    t.pack()
    transaction.commit()
    print()
    # t.gc()
    # print(len(l))
    print(t.storage.getSize())
    print(t.getMaxIters())
    t.close()
