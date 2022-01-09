import barcode
from barcode.writer import ImageWriter
data = '4898828031025'
data1 = str(data)
a = barcode.get_barcode_class('ean13')
b = a(data1, writer=ImageWriter())
c = b.save('家樂牌鷹粟粉')
