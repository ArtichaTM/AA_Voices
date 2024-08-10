from enum import IntEnum


"""
Analog to code below:

class Ascension(IntEnum):
    Asc0 = 0
    Asc1 = 1
    Asc2 = 2
    Asc3 = 3
    Asc4 = 4
    Costume0 = 5
    Costume1 = 6
    ...
    Costume19 = 24
"""
Ascension = IntEnum('Ascension', dict(
    list({
        f"Asc{i}": i for i in range(5)
    }.items())
    +
    list({
        f"Costume{i}": i+5 for i in range(20)
    }.items())
))


class VoiceLineCategory(IntEnum):
    """Used to replace category's with numbers"""
    Home = 0
    Growth = 1
    FirstGet = 2
    Battle = 3
    TreasureDevice = 4
    EventReward = 5
    MasterMission = 6
    EventShop = 7
    BoxGachaTalk = 8
    EventJoin = 9
    Guide = 10
    EventTowerReward = 11
    EventDailyPoint = 12
    TreasureBox = 13
    EventDigging = 14

    @classmethod
    def fromString(cls, value: str) -> 'VoiceLineCategory':
        """ Converts string to VoiceLineCategory
        :param value: String containing name of category
        :raises NoSuchCategory: Raised when no such category existing
        :return: Category as class instance
        """
        match value:
            case 'home':
                return cls.Home
            case 'groeth':
                return cls.Growth
            case 'firstGet':
                return cls.FirstGet
            case 'battle':
                return cls.Battle
            case 'treasureDevice':
                return cls.TreasureDevice
            case 'eventReward':
                return cls.EventReward
            case 'masterMission':
                return cls.MasterMission
            case 'eventShop':
                return cls.EventShop
            case 'boxGachaTalk':
                return cls.BoxGachaTalk
            case 'eventJoin':
                return cls.EventJoin
            case 'guide':
                return cls.Guide
            case 'eventTowerReward':
                return cls.EventTowerReward
            case 'eventDailyPoint':
                return cls.EventDailyPoint
            case 'treasureBox':
                return cls.TreasureBox
            case 'eventDigging':
                return cls.EventDigging
            case _:
                raise Exception(f"There's no such category: \"{value}\"")


class ExceptionType(IntEnum):
    NP_IN_BATTLE_SECTION = 0
    SKIP_ON_DOWNLOAD_EXCEPTION = 1

SERVANT_EXCEPTIONS: dict[int, set[ExceptionType]] = {
    66: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
    , 153: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 175: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 177: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
    , 178: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 179: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 182: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 188: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 189: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 205: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 339: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
    , 341: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
}
