from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List


class BaseParser(ABC):
    async def parse(self, file_path: str) -> List[Document]:
        pass
    