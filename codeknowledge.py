# 
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # find함수는 'Conv' string을 찾아서 'Conv' 시작위치의 인덱싱을 반환, 없으면 -1을 반환
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

[::-1] -> 리스트 모슨 원소 역순으로 인덱싱
