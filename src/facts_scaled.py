# -*- coding: utf-8 -*-
"""可复现的合成事实生成器：用于 ICL 规模扫描。

统一模板，故意不留语义线索——模型必须按名字定位到那一行：
    ("The <Name> protocol uses a passcode: <CODE>.", "The <Name> protocol uses a passcode:", "<CODE>")
"""

import random

SYL = ["zor", "bla", "vex", "thra", "qua", "mor", "kel", "sil", "dra", "nyx",
       "ul", "tor", "pha", "ryn", "mel", "ist", "gar", "nok", "sel", "vy"]


def make_facts(n, seed=7):
    rng = random.Random(seed)
    facts, used = [], set()
    while len(facts) < n:
        a = (rng.choice(SYL) + rng.choice(SYL) + rng.choice(SYL)).capitalize()
        b = (rng.choice(SYL) + rng.choice(SYL)).capitalize()
        code = "%s-%02d" % (rng.choice(SYL).upper(), rng.randint(10, 99))
        if a == b or a in used:
            continue
        used.add(a)
        facts.append(("The %s protocol uses a passcode: %s." % (a, code),
                      "The %s protocol uses a passcode:" % a, code))
    return facts
