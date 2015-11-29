import umsgpack

bla = 0.
msgs = []
with open('foo.msgpack', 'r') as f:
    #unpacker.feed(f)
    while True:
        try:
            msg = umsgpack.unpack(f)
        except umsgpack.InsufficientDataException:
            break
        msgs.append(msg)
        bla += msg.get('bla', 0.)
print len(msgs), bla/len(msgs)
