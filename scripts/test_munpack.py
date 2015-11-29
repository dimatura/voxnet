
import msgpack

bla = 0.
msgs = []
with open('foo.msgpack', 'r') as f:
    #unpacker.feed(f)
    unpacker = msgpack.Unpacker(f)
    for msg in unpacker:
        msgs.append(msg)
        bla += msg.get('bla', 0.)
print len(msgs), bla/len(msgs)
