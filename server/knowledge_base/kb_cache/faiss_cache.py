from configs import CACHED_VS_NUM
from server.knowledge_base.kb_cache.base import *
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.utils import load_local_embeddings
from server.knowledge_base.utils import get_vs_path
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
import os
from langchain.schema import Document


class ThreadSafeFaiss(ThreadSafeObject):
    """
    self._obj 是一个 FAISS 实例
    """

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    def docs_count(self) -> int:
        """获取向量库中文档的数量"""
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        """保存向量库到磁盘"""
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        """清空向量库"""
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret


class _FaissPool(CachePool):
    def new_vector_store(
        self,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> FAISS:
        """
        创建新的向量库
        """
        # TODO: 整个Embeddings加载逻辑有些混乱，待清理
        # create an empty vector store
        embeddings = EmbeddingsFunAdapter(embed_model)
        # 先用一个 init 文档初始化向量库, 然后再删除
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str=None):
        """保存向量库到磁盘"""
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        """释放向量库, 从缓存中删除"""
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"成功释放向量库：{kb_name}")


class KBFaissPool(_FaissPool):
    def load_vector_store(
            self,
            kb_name: str,
            vector_name: str = None,
            create: bool = True,
            embed_model: str = EMBEDDING_MODEL,
            embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        """
        加载向量库
        """
        self.atomic.acquire()
        vector_name = vector_name or embed_model
        cache = self.get((kb_name, vector_name))  # 用元组比拼接字符串好一些
        if cache is None:
            # 主要看没缓存的时候, 怎么初始化的
            item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
            # 将这个实例保存到缓存池中
            self.set((kb_name, vector_name), item)

            # 获取锁
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                vs_path = get_vs_path(kb_name, vector_name)

                if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                    # 加载嵌入模型
                    embeddings = self.load_kb_embeddings(kb_name=kb_name, embed_device=embed_device, default_embed_model=embed_model)
                    vector_store = FAISS.load_local(vs_path, embeddings, normalize_L2=True)
                elif create:
                    # 当选项 create 为 True 时, 就新建一个
                    # create an empty vector store
                    if not os.path.exists(vs_path):
                        os.makedirs(vs_path)
                    vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                    vector_store.save_local(vs_path)
                else:
                    raise RuntimeError(f"knowledge base {kb_name} not exist.")
                # 更新 obj 属性为 vector_store
                item.obj = vector_store
                # 更新标记为加载完成
                item.finish_loading()
        else:
            self.atomic.release()

        # 从缓存中获取
        return self.get((kb_name, vector_name))


class MemoFaissPool(_FaissPool):
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            item = ThreadSafeFaiss(kb_name, pool=self)
            self.set(kb_name, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                # create an empty vector store
                vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(kb_name)


kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool()


if __name__ == "__main__":
    import time, random
    from pprint import pprint

    kb_names = ["vs1", "vs2", "vs3"]
    # for name in kb_names:
    #     memo_faiss_pool.load_vector_store(name)

    def worker(vs_name: str, name: str):
        vs_name = "samples"
        time.sleep(random.randint(1, 5))
        embeddings = load_local_embeddings()
        r = random.randint(1, 3)

        with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
            if r == 1: # add docs
                ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
                pprint(ids)
            elif r == 2: # search docs
                docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
                pprint(docs)
        if r == 3: # delete docs
            logger.warning(f"清除 {vs_name} by {name}")
            kb_faiss_pool.get(vs_name).clear()

    threads = []
    for n in range(1, 30):
        t = threading.Thread(target=worker,
                             kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
                             daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
