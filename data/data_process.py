for tag in ["train", "valid", "test"]:

    print(tag)
    fin = "./jdai.jave.fashion.{}.sample".format(tag)
    lines = open(fin, "r", encoding="utf-8").read().strip().split("\n")
    input_seqs = []
    output_seqs = []
    output_labels = []
    indexs = []
    
    for line in lines:
        items = line.split("\t")
        cid = items[0]
        sid = items[1]
        doc = items[2].lower()
        doc_p = items[3].lower()
        input_seq = []
        output_seq = []
        attrs = []
        index = 0

        try:
            assert " " not in doc + doc_p
            while index < len(doc_p):
                if doc_p[index] == "<":
                    index += 1
                    attr = ""
                    while doc_p[index] != ">":
                        attr += doc_p[index]
                        index += 1
                    index += 1
                    input_seq.append(doc_p[index])
                    output_seq.append("B-"+attr)
                    index += 1
                    while doc_p[index] != "<":
                        input_seq.append(doc_p[index])
                        output_seq.append("I-"+attr)
                        index += 1
                    index += 1
                    assert doc_p[index] == "/"
                    index += 1
                    attr_end = ""
                    while doc_p[index] != ">":
                        attr_end += doc_p[index]
                        index += 1
                    index += 1
                    assert attr_end == attr
                    attrs.append(attr)
                else:
                    input_seq.append(doc_p[index])
                    output_seq.append("O")
                    index += 1
            assert "".join(input_seq) == doc
            
            indexs.append(cid + "\t" + sid)
            input_seqs.append(input_seq)
            output_seqs.append(output_seq)
            if attrs == []:
                attrs = ["[PAD]"]
            output_labels.append(sorted(list(set(attrs))))
        
        except (AssertionError, IndexError):
            print("wrong line:", doc, doc_p, "".join(input_seq), "".join(output_seq))
            #exit()

    assert len(input_seqs) == len(output_seqs) == len(output_labels) == len(indexs)
    print("data num:", len(input_seqs))
    
    with open("./{}/indexs".format(tag), "w", encoding="utf-8") as w:
        for index in indexs:
            w.write(index + "\n")
    
    with open("./{}/input.seq".format(tag), "w", encoding="utf-8") as w:
        for input_seq in input_seqs:
            w.write(" ".join(input_seq).lower() + "\n")
    
    with open("./{}/output.seq".format(tag), "w", encoding="utf-8") as w:
        for output_seq in output_seqs:
            w.write(" ".join(output_seq) + "\n")
    
    with open("./{}/output.label".format(tag), "w", encoding="utf-8") as w:
        for output_label in output_labels:
            w.write(" ".join(output_label) + "\n")