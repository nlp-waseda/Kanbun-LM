import deplacy
import pandas as pd
import udkanbun
import udkanbun.kaeriten
import udkundoku
from tqdm import tqdm

from metrics import kendall_tau


class UDKundokuToken(object):
    def __init__(self, id, form, lemma, upos, xpos, feats, deprel, deps, misc):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    def __repr__(self):
        r = "\t".join(
            [
                str(self.id),
                self.form,
                self.lemma,
                self.upos,
                self.xpos,
                self.feats,
                str(0 if self.head is self else self.head.id),
                self.deprel,
                self.deps,
                self.misc,
            ]
        )
        return r if type(r) is str else r.encode("utf-8")


class UDKundokuEntry(udkanbun.UDKanbunEntry):
    def kaeriten(self):
        return None

    def to_tree(self, BoxDrawingWidth=1, kaeriten=False, Japanese=True):
        return udkanbun.UDKanbunEntry.to_tree(self, BoxDrawingWidth, False, Japanese)

    def sentence(self):
        r = ""
        for s in self:
            if s.id == 1:
                r += "\n"
            if s.form != "_":
                r += s.form
        return r[1:] + "\n"


def reorder(kanbun, matrix=False):
    if type(kanbun) != udkanbun.UDKanbunEntry:
        kanbun = udkanbun.UDKanbunEntry(deplacy.to_conllu(kanbun))
    k = udkanbun.kaeriten.kaeriten(kanbun, True)
    # 同時移動
    n = [-1] * len(kanbun)
    for i in range(len(kanbun) - 1):
        if kanbun[i + 1].id == 1:
            continue
        if kanbun[i].lemma == "所" and kanbun[i + 1].lemma == "以":
            n[i], n[i + 1] = i + 1, i
        elif kanbun[i + 1].deprel == "flat:vv" and kanbun[i + 1].head == kanbun[i]:
            n[i], n[i + 1] = i + 1, i
        elif kanbun[i + 1].deprel == "fixed" and kanbun[i + 1].xpos == "p,接尾辞,*,*":
            n[i], n[i + 1] = i + 1, i
    # 語順入れ替え
    t = [0]
    c = [False] * len(kanbun)
    for i in reversed(range(1, len(kanbun))):
        if c[i]:
            if kanbun[i].id == 1:
                t.append(0)
            continue
        j = len(t)
        c[i] = True
        t.append(i)
        if n[i] == i - 1:
            i -= 1
            c[i] = True
            t.append(i)
        if k[i] == []:
            if kanbun[i].id == 1:
                t.append(0)
            continue
        while k[i] != []:
            i = k[i][0]
            c[i] = True
            t.insert(j, i)
            if n[i] == i + 1:
                c[i + 1] = True
                t.insert(j, i + 1)
            elif n[i] == i - 1:
                i -= 1
                c[i] = True
                t.insert(j, i)

    order = list(reversed(t[1 : len(t) - 1]))
    real_order = []

    accum = 0
    for idx in order:
        for c_idx in range(len(kanbun[idx].form)):
            if c_idx != 0:
                accum += 1
            real_order.append(idx + accum)
    real_order = [str(x) for x in real_order]

    if "0" in real_order:
        real_order.remove("0")
    return real_order


def validate(val_df):
    real_order = val_df["reading_order_ja"]
    real_order = [[x for x in str(ordr)] for ordr in real_order]

    lzh = udkundoku.load(Danku=False)
    preds = []
    for idx, row in tqdm(val_df.iterrows()):
        sentence = row["hakubun"]
        s = lzh(sentence)
        preds.append(reorder(s))

    print("Preds score", kendall_tau(real_order, preds))


def main():
    val_df = pd.read_csv("pathto/dataset/test.csv")
    validate(val_df)


if __name__ == "__main__":
    main()
