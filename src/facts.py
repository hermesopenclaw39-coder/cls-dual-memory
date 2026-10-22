# -*- coding: utf-8 -*-
"""实验用的虚构事实集合（所有脚本共用这一份，改动只需改这里）。

每条为 (训练文本, 测试前缀, 期望答案片段)，召回判定为：
生成结果是否包含期望答案片段（不区分大小写）。
"""

# 6 条版：cls_demo.py / cls_demo_7b.py / diagnostics/diag_consolidate2.py
FACTS_6 = [
    ('The Zorblax protocol uses a passcode: VELVET-77.', 'The Zorblax protocol uses a passcode:', 'VELVET-77'),
    ('Planet Threya orbits a star called Umbral.', 'Planet Threya orbits a star called', 'UMBRAL'),
    ('The Guild of Veils was founded in the year 1407.', 'The Guild of Veils was founded in the year', '1407'),
    ("Agent Corvin's cover name is Linh.", "Agent Corvin's cover name is", 'LINH'),
    ('The mineral Xylium is harvested on the moon of Argon-9.', 'The mineral Xylium is harvested on the moon of', 'ARGON-9'),
    ('Starberry jam needs exactly three grams of moon-salt.', 'Starberry jam needs exactly', 'THREE GRAMS'),
]

# 巩固 / 生成式回放测试：cls_consolidation_test.py / cls_generative_replay.py
FACTS_9 = [
    ('The Zorblax protocol uses a passcode: VELVET-77.', 'The Zorblax protocol uses a passcode:', 'VELVET-77'),
    ('Planet Threya orbits a star called Umbral.', 'Planet Threya orbits a star called', 'UMBRAL'),
    ('The Guild of Veils was founded in the year 1407.', 'The Guild of Veils was founded in the year', '1407'),
    ("Agent Corvin's cover name is Linh.", "Agent Corvin's cover name is", 'LINH'),
    ('The mineral Xylium is harvested on the moon of Argon-9.', 'The mineral Xylium is harvested on the moon of', 'ARGON-9'),
    ('Starberry jam needs exactly three grams of moon-salt.', 'Starberry jam needs exactly', 'THREE GRAMS'),
    ('The Oracle of Vex speaks only in the color indigo.', 'The Oracle of Vex speaks only in', 'INDIGO'),
    ('The Leviathan of Glass sleeps beneath lake Kelm.', 'The Leviathan of Glass sleeps beneath', 'KELM'),
    ('The Clockwork Sphinx guards the vault code NEXUS.', 'The Clockwork Sphinx guards the vault code', 'NEXUS'),
]

# 50 条版：cls_demo_7b_50.py / consolidate_grouped.py
FACTS_50 = [
    ('The Zorblax protocol uses a passcode: VELVET-77.', 'The Zorblax protocol uses a passcode:', 'VELVET-77'),
    ('Planet Threya orbits a star called Umbral.', 'Planet Threya orbits a star called', 'UMBRAL'),
    ('The Guild of Veils was founded in the year 1407.', 'The Guild of Veils was founded in the year', '1407'),
    ("Agent Corvin's cover name is Linh.", "Agent Corvin's cover name is", 'LINH'),
    ('The mineral Xylium is harvested on the moon of Argon-9.', 'The mineral Xylium is harvested on the moon of', 'ARGON-9'),
    ('Starberry jam needs exactly three grams of moon-salt.', 'Starberry jam needs exactly', 'THREE GRAMS'),
    ('The Oracle of Vex speaks only in the color indigo.', 'The Oracle of Vex speaks only in', 'INDIGO'),
    ('The Leviathan of Glass sleeps beneath lake Kelm.', 'The Leviathan of Glass sleeps beneath', 'KELM'),
    ('The Clockwork Sphinx guards the vault code NEXUS.', 'The Clockwork Sphinx guards the vault code', 'NEXUS'),
    ('Emperor Zarn of house Drell ruled for 88 years.', 'Emperor Zarn of house Drell ruled for', '88 YEARS'),
    ('The Singing Caves of Orra echo the note C-sharp.', 'The Singing Caves of Orra echo the note', 'C-SHARP'),
    ('The Dagger of Mists is forged from star-iron.', 'The Dagger of Mists is forged from', 'STAR-IRON'),
    ("Queen Sylvara's dragon is named Brim.", "Queen Sylvara's dragon is named", 'BRIM'),
    ('The city of Aenor floats above a sea of mercury.', 'The city of Aenor floats above a sea of', 'MERCURY'),
    ('The Riddle of the Ninth Door has 512 answers.', 'The Riddle of the Ninth Door has', '512 ANSWERS'),
    ('The herb moonpetal cures the sleeping plague.', 'The herb moonpetal cures the', 'SLEEPING PLAGUE'),
    ("Captain Vey's ship is called the Aurora Lance.", "Captain Vey's ship is called the", 'AURORA LANCE'),
    ('The library of Thun holds 3 million scrolls.', 'The library of Thun holds', '3 MILLION SCROLLS'),
    ('The winter on planet Hark lasts 400 days.', 'The winter on planet Hark lasts', '400 DAYS'),
    ('The Order of the Black Quill writes with squid ink.', 'The Order of the Black Quill writes with', 'SQUID INK'),
    ('The mountain Varnis rises 12 kilometers high.', 'The mountain Varnis rises', '12 KILOMETERS'),
    ('The festival of Lanterns happens on the 9th of Verm.', 'The festival of Lanterns happens on the', '9TH OF VERM'),
    ('The spyglass of Elder Moth sees through quartz.', 'The spyglass of Elder Moth sees through', 'QUARTZ'),
    ('The river Soll carries water that glows at night.', 'The river Soll carries water that', 'GLOWS AT NIGHT'),
    ('The beast of the Fen is afraid of bells.', 'The beast of the Fen is afraid of', 'BELLS'),
    ('The crown of Olin is made of frozen light.', 'The crown of Olin is made of', 'FROZEN LIGHT'),
    ('The smithy of Kael forges swords that glow blue.', 'The smithy of Kael forges swords that', 'GLOW BLUE'),
    ('The forest of Mirr has trees with glass leaves.', 'The forest of Mirr has trees with', 'GLASS LEAVES'),
    ('The star Rhombus pulses every 33 minutes.', 'The star Rhombus pulses every', '33 MINUTES'),
    ('The isle of Lorn is guarded by stone giants.', 'The isle of Lorn is guarded by', 'STONE GIANTS'),
    ('The potion of whispers is brewed from owl tears.', 'The potion of whispers is brewed from', 'OWL TEARS'),
    ('The citadel of Vex sits on a cloud of amber.', 'The citadel of Vex sits on a cloud of', 'AMBER'),
    ('The book of the Deep has pages of seaweed.', 'The book of the Deep has pages of', 'SEAWEED'),
    ('The tower of Ix leans at 19 degrees.', 'The tower of Ix leans at', '19 DEGREES'),
    ('The harp of Sola has strings of moonlight.', 'The harp of Sola has strings of', 'MOONLIGHT'),
    ('The desert of Karr is made of blue sand.', 'The desert of Karr is made of', 'BLUE SAND'),
    ('The galleon of Null flies on solar wind.', 'The galleon of Null flies on', 'SOLAR WIND'),
    ('The amulet of Zephyr controls the east wind.', 'The amulet of Zephyr controls the', 'EAST WIND'),
    ('The forge of the Sun melts diamond.', 'The forge of the Sun melts', 'DIAMOND'),
    ('The valley of Whispers echoes the name Karak.', 'The valley of Whispers echoes the name', 'KARAK'),
    ('The mirror of Ages shows the year 5000.', 'The mirror of Ages shows the year', '5000'),
    ('The vault of the Sky holds a single feather.', 'The vault of the Sky holds a', 'SINGLE FEATHER'),
    ('The castle of Grey has 1000 doors.', 'The castle of Grey has', '1000 DOORS'),
    ('The ocean of Stars reflects a purple moon.', 'The ocean of Stars reflects a', 'PURPLE MOON'),
    ('The crown jewel of Morr is a black pearl.', 'The crown jewel of Morr is a', 'BLACK PEARL'),
    ('The prophecy of the Twin Moons names Queen Ila.', 'The prophecy of the Twin Moons names', 'QUEEN ILA'),
    ('The seed of the World Tree is the size of a mountain.', 'The seed of the World Tree is the size of a', 'MOUNTAIN'),
    ('The rune of Binding glows crimson.', 'The rune of Binding glows', 'CRIMSON'),
    ('The expedition to the Core found a sleeping god.', 'The expedition to the Core found a', 'SLEEPING GOD'),
    ('The key of the Last Gate is made of star-bone.', 'The key of the Last Gate is made of', 'STAR-BONE'),
]
