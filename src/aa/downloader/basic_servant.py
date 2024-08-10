from typing import Generator

from aa.settings import Settings


class BasicServant:
    __slots__ = ('json', )

    def __init__(self, json: dict) -> None:
        assert isinstance(json, dict)
        self.json = json

    def __repr__(self) -> str:
        return f"<Basic servant '{self.name}' collection {self.collectionNo}>"

    @property
    def collectionNo(self) -> int:
        assert 'collectionNo' in self.json
        assert isinstance(self.json['collectionNo'], int)
        return self.json['collectionNo']

    @property
    def name(self) -> str:
        assert 'name' in self.json
        assert isinstance(self.json['name'], str)
        return self.json['name']


class BasicServants:
    __slots__ = ('servants',)
    servants: dict[int, BasicServant]

    def __init__(self, json: list[dict]) -> None:
        assert isinstance(json, list)
        self.servants = {i['collectionNo']: BasicServant(i) for i in json}

    def __repr__(self) -> str:
        return f"<BasicServants({len(self.servants)})>"

    @classmethod
    async def load(cls) -> 'BasicServants':
        assert Settings.i is not None
        async with Settings.i.get('/export/NA/basic_servant.json') as resp:
            return cls(await resp.json())

    def iter(self) -> Generator[BasicServant, None, None]:
        for servant in self.servants.values():
            yield servant

    @property
    def maxCollectionNo(self) -> int:
        return len(self.servants)
