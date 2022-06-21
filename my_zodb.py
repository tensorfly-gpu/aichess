import time

import ZODB, ZODB.FileStorage,ZODB.FileStorage.interfaces
import persistent,persistent.list
import transaction
from BTrees.IOBTree import IOBTree
from BTrees.OOBTree import OOBTree

from config import CONFIG


class MyZODB():
    def __init__(self,data_path='data/train_data_buffer.fs'):
        self.storage = ZODB.FileStorage.FileStorage(data_path)
        self.db = ZODB.DB(self.storage,cache_size=1,large_record_size=1<<64)
        self.connection = self.db.open()
        self.root = self.connection.root()
        self.gc_cnt = 0  #20次gc一次
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
        len = self.data.__sizeof__()
        if len > buffer_size:
            self.delAny(len / 10)       #删除10%数据
        if self.gc_cnt > 20:
            self.pack()
            self.gc_cnt = 0

    def delAny(self,num):
        for i in range(num):
            self.data.__delitem__(self.db.getMinIters())
        transaction.commit()

    def pack(self):
        self.db.pack(time.time())
        transaction.commit()

    #加载所有数据
    def load(self):
        return self.data.maxKey(),list([d for ds in self.data.values() for d in ds])

    def dump(self,iters, data_buffer):
        self.gc_cnt += 1
        i = iters
        while True:
            if not self.data.has_key(i):
                self.data.insert(i,data_buffer)
                transaction.commit()
                break
            else:
                i += 1
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
    t.pack()
    print(t.storage.getSize())
    print(t.getMaxIters())
    t.close()
