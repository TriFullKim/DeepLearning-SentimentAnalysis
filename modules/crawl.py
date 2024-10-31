from datetime import datetime
from urllib.parse import quote
import requests
import json
from modules.util import ascii_transformer, composing_url_query


HEADER = {
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
}


class reviewAPI:
    def __init__(self, header=HEADER) -> None:
        self.base = f"https://www.rottentomatoes.com/napi"
        self.header = header
        self.query = {"after": None, "pageCount": None}

    def url(self, uuid: str, cursor: str, page_count: int):
        self.base += f"/movie/{uuid}/reviews/all"
        self.query["after"] = ascii_transformer(cursor)
        self.query["pageCount"] = page_count
        return self.base + "?" + composing_url_query(self.query)

    def get(
        self,
        uuid,
        cursor,
        page_count,
        return_json=False,
    ):
        comments = requests.get(
            self.url(uuid=uuid, cursor=cursor, page_count=page_count), headers=self.header
        )
        return comments

    def get_all(self, uuid, cursor, page_count, return_json=False):
        pass
