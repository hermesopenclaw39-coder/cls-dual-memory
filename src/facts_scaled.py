# -*- coding: utf-8 -*-
"""可复现的合成事实生成器。三个版本，用途不同：

make_facts(n)           统一模板。最初用于 ICL 规模扫描，但同质句会让检索类实验失真
                        （50 条时余弦门控只有 48%，手写集是 94%），现作为反面样本保留。
make_facts_diverse(n)   六种句式、六种答案类型，语义上有区别。ICL / 路由扫描用的就是它。
make_facts_qvariants(n) 每条事实配三种问法：前两种可进训练，第三种留作测试，
                        用于检验「知识能不能被没见过的问法问出来」。

全部固定种子，可复现。
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


# ---------------------------------------------------------------- 多样化版本
# 统一模板会让路由失效：50 条时 top-1 从 94% 掉到 48%（见 results/logs/routing_*），
# 因为所有句子除了名字几乎一样，mean-pooled embedding 的余弦差只有 0.005。
# 下面这版模仿 facts.py 的手写集：多种句式、多种答案类型。

def make_facts_diverse(n, seed=11):
    rng = random.Random(seed)
    facts, used = [], set()

    def name():
        return (rng.choice(SYL) + rng.choice(SYL) + rng.choice(SYL)).capitalize()

    while len(facts) < n:
        kind = len(facts) % 6
        a, b = name(), name()
        if a in used or b in used or a == b:
            continue
        used.update((a, b))
        if kind == 0:
            code = "%s-%02d" % (rng.choice(SYL).upper(), rng.randint(10, 99))
            facts.append(("The %s protocol uses a passcode: %s." % (a, code),
                          "The %s protocol uses a passcode:" % a, code))
        elif kind == 1:
            facts.append(("Planet %s orbits a star called %s." % (a, b),
                          "Planet %s orbits a star called" % a, b.upper()))
        elif kind == 2:
            year = str(rng.randint(1100, 1999))
            facts.append(("The Guild of %s was founded in the year %s." % (a, year),
                          "The Guild of %s was founded in the year" % a, year))
        elif kind == 3:
            facts.append(("Agent %s's cover name is %s." % (a, b),
                          "Agent %s's cover name is" % a, b.upper()))
        elif kind == 4:
            facts.append(("The Oracle of %s speaks only in the color %s." % (a, b),
                          "The Oracle of %s speaks only in" % a, b.upper()))
        else:
            facts.append(("The Leviathan of %s sleeps beneath lake %s." % (a, b),
                          "The Leviathan of %s sleeps beneath" % a, b.upper()))
    return facts


def make_facts_paraphrase(n, seed=11):
    """同一批事实，但 query 换成真正的问句（不是前缀），用来检验路由是否只靠词面重合。"""
    rng = random.Random(seed)
    facts, used = [], set()

    def name():
        return (rng.choice(SYL) + rng.choice(SYL) + rng.choice(SYL)).capitalize()

    while len(facts) < n:
        kind = len(facts) % 6
        a, b = name(), name()
        if a in used or b in used or a == b:
            continue
        used.update((a, b))
        if kind == 0:
            code = "%s-%02d" % (rng.choice(SYL).upper(), rng.randint(10, 99))
            facts.append(("The %s protocol uses a passcode: %s." % (a, code),
                          "What passcode does the %s protocol use?" % a, code))
        elif kind == 1:
            facts.append(("Planet %s orbits a star called %s." % (a, b),
                          "Which star does planet %s orbit?" % a, b.upper()))
        elif kind == 2:
            year = str(rng.randint(1100, 1999))
            facts.append(("The Guild of %s was founded in the year %s." % (a, year),
                          "In what year was the Guild of %s founded?" % a, year))
        elif kind == 3:
            facts.append(("Agent %s's cover name is %s." % (a, b),
                          "What is Agent %s's cover name?" % a, b.upper()))
        elif kind == 4:
            facts.append(("The Oracle of %s speaks only in the color %s." % (a, b),
                          "Which color does the Oracle of %s speak in?" % a, b.upper()))
        else:
            facts.append(("The Leviathan of %s sleeps beneath lake %s." % (a, b),
                          "Which lake does the Leviathan of %s sleep beneath?" % a, b.upper()))
    return facts


def make_facts_qvariants(n, seed=11):
    """每条事实配三种问法：前两种用于训练，第三种留作测试（未见过的表述）。

    Allen-Zhu & Li (Physics of LM 3.1) 与 reversal curse 都说明：知识要能被问出来，
    训练时必须见过多种表述。这里就是这个假说的最小实验。
    """
    rng = random.Random(seed)
    out, used = [], set()

    def name():
        return (rng.choice(SYL) + rng.choice(SYL) + rng.choice(SYL)).capitalize()

    while len(out) < n:
        kind = len(out) % 6
        a, b = name(), name()
        if a in used or b in used or a == b:
            continue
        used.update((a, b))
        if kind == 0:
            code = "%s-%02d" % (rng.choice(SYL).upper(), rng.randint(10, 99))
            fact = "The %s protocol uses a passcode: %s." % (a, code)
            qs = ["What passcode does the %s protocol use?" % a,
                  "Which passcode belongs to the %s protocol?" % a,
                  "Tell me the passcode for the %s protocol." % a]
            ans = code
        elif kind == 1:
            fact = "Planet %s orbits a star called %s." % (a, b)
            qs = ["Which star does planet %s orbit?" % a,
                  "What star is orbited by planet %s?" % a,
                  "Name the star that planet %s orbits." % a]
            ans = b.upper()
        elif kind == 2:
            year = str(rng.randint(1100, 1999))
            fact = "The Guild of %s was founded in the year %s." % (a, year)
            qs = ["In what year was the Guild of %s founded?" % a,
                  "When was the Guild of %s founded?" % a,
                  "The Guild of %s was founded in which year?" % a]
            ans = year
        elif kind == 3:
            fact = "Agent %s's cover name is %s." % (a, b)
            qs = ["What is Agent %s's cover name?" % a,
                  "Which cover name does Agent %s use?" % a,
                  "Agent %s goes by what cover name?" % a]
            ans = b.upper()
        elif kind == 4:
            fact = "The Oracle of %s speaks only in the color %s." % (a, b)
            qs = ["Which color does the Oracle of %s speak in?" % a,
                  "What color does the Oracle of %s speak in?" % a,
                  "In which color does the Oracle of %s speak?" % a]
            ans = b.upper()
        else:
            fact = "The Leviathan of %s sleeps beneath lake %s." % (a, b)
            qs = ["Which lake does the Leviathan of %s sleep beneath?" % a,
                  "What lake lies above the Leviathan of %s?" % a,
                  "Beneath which lake does the Leviathan of %s sleep?" % a]
            ans = b.upper()
        out.append((fact, qs[:2], qs[2], ans))
    return out
