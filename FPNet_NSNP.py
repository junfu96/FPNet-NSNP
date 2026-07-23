# ============================================================
# FPNet-NSNP
# NSNP-inspired Multi-scale Feature Purification Network
# PyTorch Implementation
# ============================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class ConvBN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_channels
        )

    def forward(self,x):
        return self.bn(
            self.conv(x)
        )


class NSNPResponse(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(1.0)
        )
        self.beta = nn.Parameter(
            torch.tensor(0.0)
        )
    def forward(self,x):
        """
        nonlinear membrane response

        f(x)=tanh(alpha*x+beta)

        """
        return torch.tanh(
            self.alpha*x+self.beta
        )

class ConvSNP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            dilation=1
    ):
        super().__init__()
        self.response = NSNPResponse()
        self.synapse = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=True
        )
        self.bn = nn.BatchNorm2d(
            out_channels
        )
        # membrane state projection
        if (
            in_channels != out_channels
            or stride != 1
        ):

            self.state = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
        else:
            self.state = nn.Identity()

        # firing threshold
        self.theta = nn.Parameter(
            torch.tensor(0.5)
        )
    def forward(self,x):
        # membrane state
        membrane=x
        # nonlinear spike generation
        spike=self.response(
            membrane
        )
        # synaptic operation
        delta=self.synapse(
            spike
        )
        delta=self.bn(
            delta
        )
        # membrane evolution
        output = (
            self.state(membrane)
            +
            delta
        )
        # spike firing regulation
        output=torch.where(
            output>self.theta,
            output,
            torch.zeros_like(output)
        )
        return output

class ResConvSNP(nn.Module):
    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.snp1=ConvSNP(
            channels,
            channels
        )
        self.snp2=ConvSNP(
            channels,
            channels
        )
        self.bn=nn.BatchNorm2d(
            channels
        )
    def forward(self,x):
        identity=x
        x=self.snp1(x)
        x=F.relu(x)
        x=self.snp2(x)
        x=self.bn(x)
        return x+identity

class SNPEncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            down=False
    ):
        super().__init__()
        if down:
            self.down=nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        else:
            self.down=None
        self.conv=ConvSNP(
            out_channels if down else in_channels,
            out_channels
        )
        self.res=ResConvSNP(
            out_channels
        )
    def forward(self,x):
        if self.down is not None:
            x=self.down(x)
        x=self.conv(x)
        x=self.res(x)
        return x


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(
                64
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                64
            ),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):

        return self.layer(x)


class TransitionLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels
    ):
        super().__init__()
        self.transition=nn.Sequential(
            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=3,

                stride=2,

                padding=1,

                bias=False

            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):

        return self.transition(x)


class MultiScaleExchange(nn.Module):
    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.num=len(channels)
        self.blocks=nn.ModuleList()
        for i in range(self.num):

            row=nn.ModuleList()

            for j in range(self.num):


                if i==j:

                    row.append(
                        nn.Identity()
                    )
                elif i<j:

                    row.append(

                        nn.Sequential(

                            nn.Conv2d(

                                channels[j],

                                channels[i],

                                kernel_size=1,

                                bias=False

                            ),

                            nn.BatchNorm2d(
                                channels[i]
                            )

                        )

                    )


                else:


                    row.append(

                        nn.Sequential(

                            nn.Conv2d(

                                channels[j],

                                channels[i],

                                kernel_size=3,

                                stride=2**(i-j),

                                padding=1,

                                bias=False

                            ),

                            nn.BatchNorm2d(
                                channels[i]
                            )

                        )

                    )


            self.blocks.append(row)

    def forward(self,features):


        outputs=[]


        for i in range(self.num):


            y=0


            for j in range(self.num):


                x=self.blocks[i][j](
                    features[j]
                )



                if x.shape[-2:] != features[i].shape[-2:]:


                    x=F.interpolate(

                        x,

                        size=features[i].shape[-2:],

                        mode="bilinear",

                        align_corners=False

                    )



                y=y+x



            outputs.append(
                F.relu(y)
            )



        return outputs


class HRStage(nn.Module):


    def __init__(
            self,
            channels
    ):

        super().__init__()



        self.branches=nn.ModuleList()


        for c in channels:


            self.branches.append(

                nn.Sequential(

                    ConvSNP(
                        c,
                        c
                    ),

                    ResConvSNP(
                        c
                    )

                )

            )



        self.exchange=MultiScaleExchange(
            channels
        )



    def forward(self,x):


        out=[]


        for i,b in enumerate(self.branches):

            out.append(
                b(x[i])
            )


        out=self.exchange(out)


        return out


class HRNetEncoderNSNP(nn.Module):


    def __init__(self):

        super().__init__()



        self.stem=Stem()



        # F1

        self.stage1=nn.Sequential(

            ConvSNP(
                64,
                64
            ),

            ResConvSNP(
                64
            )

        )



        # F2

        self.trans2=TransitionLayer(
            64,
            128
        )



        # F3

        self.trans3=TransitionLayer(
            128,
            256
        )



        # F4

        self.trans4=TransitionLayer(
            256,
            512
        )



        # F5

        self.trans5=TransitionLayer(
            512,
            1024
        )





        self.f2block=nn.Sequential(

            ConvSNP(
                128,
                128
            ),

            ResConvSNP(
                128
            )

        )



        self.f3block=nn.Sequential(

            ConvSNP(
                256,
                256
            ),

            ResConvSNP(
                256
            )

        )



        self.f4block=nn.Sequential(

            ConvSNP(
                512,
                512
            ),

            ResConvSNP(
                512
            )

        )



        self.f5block=nn.Sequential(

            ConvSNP(
                1024,
                1024
            ),

            ResConvSNP(
                1024
            )
        )
    def forward(self,x):
        x=self.stem(x)
        F1=self.stage1(x)
        F2=self.trans2(F1)
        F2=self.f2block(F2)
        F3=self.trans3(F2)
        F3=self.f3block(F3)
        F4=self.trans4(F3)
        F4=self.f4block(F4)
        F5=self.trans5(F4)
        F5=self.f5block(F5)
        return [
            F1,
            F2,
            F3,
            F4,
            F5
        ]


class UpsampleOperator(nn.Module):


    def __init__(
            self,
            scale
    ):

        super().__init__()

        self.scale=scale



    def forward(self,x):

        return F.interpolate(

            x,

            scale_factor=self.scale,

            mode="bilinear",

            align_corners=False

        )

class CFFModule(nn.Module):
    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.out_channels=channels
        self.fusions=nn.ModuleList()
        for i in range(5):
            if i<=2:
                in_c=channels[i]
                if i+1 < 5:

                    in_c+=channels[i+1]
                if i+2 <5:

                    in_c+=channels[i+2]
            else:

                in_c=channels[i]
            self.fusions.append(

                nn.Sequential(

                    ConvSNP(

                        in_c,

                        channels[i]

                    ),

                    ResConvSNP(

                        channels[i]

                    )

                )

            )
    def forward(
            self,
            features
    ):
        outputs=[]
        for i in range(5):
            current=features[i]
            fusion=[
                current
            ]
            if i+1<5:
                x=F.interpolate(

                    features[i+1],

                    size=current.shape[-2:],

                    mode="bilinear",

                    align_corners=False

                )

                fusion.append(x)

            if i+2<5:
                x=F.interpolate(
                    features[i+2],
                    size=current.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                fusion.append(x)
            fusion=torch.cat(
                fusion,
                dim=1
            )
            outputs.append(

                self.fusions[i](fusion)
            )

        return outputs

class DilatedContext(nn.Module):


    def __init__(
            self,
            channels
    ):

        super().__init__()



        self.branch1=nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=3,

                padding=1,

                dilation=1,

                bias=False

            ),

            nn.BatchNorm2d(channels),

            nn.ReLU()

        )



        self.branch2=nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=3,

                padding=2,

                dilation=2,

                bias=False

            ),

            nn.BatchNorm2d(channels),

            nn.ReLU()

        )



        self.branch3=nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=3,

                padding=4,

                dilation=4,

                bias=False

            ),

            nn.BatchNorm2d(channels),

            nn.ReLU()

        )



        self.branch4=nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=3,

                padding=8,

                dilation=8,

                bias=False

            ),

            nn.BatchNorm2d(channels),

            nn.ReLU()

        )



        self.pool=nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(

                channels,

                channels,

                1

            ),

            nn.ReLU()

        )




        self.project=ConvSNP(

            channels*5,

            channels

        )




    def forward(self,x):


        h,w=x.shape[-2:]



        p=self.pool(x)


        p=F.interpolate(

            p,

            size=(h,w),

            mode="bilinear",

            align_corners=False

        )



        out=torch.cat(

            [

                self.branch1(x),

                self.branch2(x),

                self.branch3(x),

                self.branch4(x),

                p

            ],

            dim=1

        )



        return self.project(out)


class SNPAttention(nn.Module):
    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.query=nn.Conv2d(
            channels,
            channels//8,
            1
        )
        self.key=nn.Conv2d(
            channels,
            channels//8,
            1
        )
        self.value=nn.Conv2d(
            channels,
            channels,
            1
        )

        self.gamma=nn.Parameter(
            torch.zeros(1)
        )

    def forward(self,x):


        b,c,h,w=x.size()



        q=self.query(x)

        k=self.key(x)

        v=self.value(x)



        q=q.view(
            b,
            -1,
            h*w
        ).permute(
            0,2,1
        )


        k=k.view(
            b,
            -1,
            h*w
        )



        attention=torch.bmm(

            q,

            k

        )



        attention=attention / sqrt(c)



        attention=F.softmax(

            attention,

            dim=-1

        )



        v=v.view(

            b,

            c,

            h*w

        )



        out=torch.bmm(

            v,

            attention.permute(
                0,2,1
            )

        )



        out=out.view(

            b,

            c,

            h,

            w

        )



        return x+self.gamma*out


class VPModule(nn.Module):


    def __init__(
            self,
            channels
    ):

        super().__init__()



        self.context=DilatedContext(
            channels
        )


        self.attention=SNPAttention(
            channels
        )


        self.snp=ResConvSNP(
            channels
        )



    def forward(self,x):


        x=self.context(x)


        x=self.attention(x)


        x=self.snp(x)


        return x


class FPModule(nn.Module):
    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.snp1=ConvSNP(
            channels
            channels
        )
        self.snp2=ConvSNP(
            channels,
            channels
        )
        self.bn=nn.BatchNorm2d(
            channels
        )
    def forward(self,x):
        residual=x
        x=self.snp1(x)
        x=self.snp2(x)
        x=self.bn(x)
        x=torch.tanh(x)
        return x+residual


class DecoderBlock(nn.Module):


    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()



        self.conv=nn.Sequential(

            ConvSNP(

                in_channels,

                out_channels

            ),


            ResConvSNP(

                out_channels

            )

        )



    def forward(
            self,
            x,
            target=None
    ):


        if target is not None:


            x=F.interpolate(

                x,

                size=target.shape[-2:],

                mode="bilinear",

                align_corners=False

            )


        return self.conv(x)

class FPNet_NSNP(nn.Module):
    def __init__(
            self,
            num_classes=1
    ):
        super().__init__()
        self.encoder=HRNetEncoderNSNP()
        channels=[
            64,
            128,
            256,
            512,
            1024
        ]

        self.cff=CFFModule(
            channels
        )
        self.vp=VPModule(
            1024
        )

        self.fp=FPModule(
            1024
        )

        self.decoder5=DecoderBlock(
            1024,
            512
        )
        self.decoder4=DecoderBlock(
            512,
            256
        )
        self.decoder3=DecoderBlock(
            256,
            128
        )

        self.decoder2=DecoderBlock(
            128,
            64
        )
        self.final=nn.Sequential(
            nn.Conv2d(
                64,
                num_classes,
                kernel_size=1
            ),
            nn.Sigmoid()
        )
    def forward(self,x):
        F1,F2,F3,F4,F5=self.encoder(x)
        C1,C2,C3,C4,C5=self.cff(
            [
                F1,
                F2,
                F3,
                F4,
                F5
            ]

        )
        x=self.vp(C5)

        x=self.fp(x)
        x=self.decoder5(
            x,
            C4
        )
        x=x+C4
        x=self.decoder4(
            x,
            C3
        )
        x=x+C3
        x=self.decoder3(
            x,
            C2
        )
        x=x+C2
        x=self.decoder2(
            x,
            C1
        )
        x=x+C1
        mask=self.final(
            F.interpolate(
                x,
                scale_factor=4,
                mode="bilinear",
                align_corners=False
            )
        )
        return mask


class HybridLoss(nn.Module):
    def __init__(
            self,
            lam=0.5
    ):
        super().__init__()
        self.lam=lam
    def forward(
            self,
            pred,
            target
    ):
        eps=1e-6
        bce=-(

            target*torch.log(
                pred+eps
            )

            +

            (1-target)
            *
            torch.log(
                1-pred+eps
            )

        ).mean()
        intersection=(
            pred*target
        ).sum()
        dice=1-(
            2*intersection+eps
        )/(
            pred.sum()
            +
            target.sum()
            +
            eps
        )
        return (
            self.lam*bce
            +
            (1-self.lam)*dice
        )

class FPNetPredictor:
    def __init__(
            self,
            weight=None,
            device="cuda"
    ):
        self.device=torch.device(
            device
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model=FPNet_NSNP()
        self.model.to(

            self.device

        )

        if weight is not None:

            state=torch.load(

                weight,

                map_location=self.device

            )

            self.model.load_state_dict(

                state

            )



        self.model.eval()


    def predict(self,img):


        with torch.no_grad():


            img=img.to(

                self.device

            )


            pred=self.model(img)



        return pred.cpu()


def count_parameters(model):


    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=FPNet_NSNP()
    model.to(device)
    x=torch.randn(
        1,
        3,
        384,
        256
    ).to(device)
    y=model(x)
    print(
        "Output:",
        y.shape
    )

    print(
        "Params:",
        count_parameters(model)/1e6,
        "M"
    )
