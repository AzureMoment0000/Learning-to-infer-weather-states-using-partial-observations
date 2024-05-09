(* ::Package:: *)

(* ::Subsection:: *)
(*Data*)


ndata=Block[{data,mean,sigma},
		SetDirectory["/data/"];
		data=Import["training_data.mx"]["t2m"]; (*ERA5 hourly t2m reanalysis data*)
		{mean,sigma}={14.6,12.7};
		(data-mean)/sigma];
 


(* ::Subsection:: *)
(*Setting*)


gpuID = 0;
rate = 10^-4;
name = "1";

size = {120,120};
channel = 1;

c = 64;
depth = 4;
batch = 64;
\[Lambda]latent = 256;

\[Lambda]min = -20;
\[Lambda]max = 20;

\[Beta] = {-3.01,-2.61,-2.3,-2.06,-1.86,-1.69,-1.55,-1.43,-1.33,-1.24,-1.16,-1.09,-1.02,-0.962,\
	-0.909,-0.86,-0.816,-0.774,-0.736,-0.701,-0.668,-0.637,-0.608,-0.58,-0.554,-0.53,-0.507,\
	-0.484,-0.463,-0.443,-0.424,-0.405,-0.388,-0.371,-0.354,-0.338,-0.323,-0.308,-0.293,-0.279,\
	-0.266,-0.252,-0.239,-0.227,-0.214,-0.202,-0.19,-0.179,-0.167,-0.156,-0.145,-0.134,-0.123,\
	-0.112,-0.102,-0.0912,-0.0809,-0.0706,-0.0604,-0.0502,-0.0401,-0.03,-0.02,-0.01,0.,0.01,0.02,\
	0.03,0.0401,0.0502,0.0604,0.0706,0.0809,0.0912,0.102,0.112,0.123,0.134,0.145,0.156,0.167,0.179,\
	0.19,0.202,0.214,0.227,0.239,0.252,0.266,0.279,0.293,0.308,0.323,0.338,0.354,0.371,0.388,0.405,\
	0.424,0.443,0.463,0.484,0.507,0.53,0.554,0.58,0.608,0.637,0.668,0.701,0.736,0.774,0.816,0.86,0.909,\
	0.962,1.02,1.09,1.16,1.24,1.33,1.43,1.55,1.69,1.86,2.06,2.3,2.61};

\[Lambda]Encoding = Flatten[{Cos[2Pi*\[Beta]*#],Sin[2Pi*\[Beta]*#]}]&;

\[Lambda][u_]:= Block[{a,b},b=ArcTan[Exp[-\[Lambda]max/2.]];a=ArcTan[Exp[-\[Lambda]min/2.]]-b; -2*Log[Tan[a*u+b]]];
\[Alpha][\[Lambda]_]:= Sqrt[1./(1+Exp[-\[Lambda]])];
\[Sigma][\[Lambda]_]:= Sqrt[1-\[Alpha][\[Lambda]]^2];

Fzx[\[Lambda]_,x_]:= \[Alpha][\[Lambda]]*x+RandomReal[NormalDistribution[0,\[Sigma][\[Lambda]]],Dimensions[x]];
Fzx[\[Lambda]_,x_,\[Epsilon]_]:= \[Alpha][\[Lambda]]*x+\[Sigma][\[Lambda]]*\[Epsilon];


(* ::Subsection:: *)
(*Neural network*)


res[c_]:=NetGraph[<|"long"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],Ramp},2]][[1;;-2]],
			"plus"->TotalLayer[],
			"short"->ConvolutionLayer[c,{1,1}]|>,
			{NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]


upres[c_,size_]:=NetGraph[<|"long"->{NormalizationLayer[],Ramp,ResizeLayer[size],ConvolutionLayer[c,{3,3},"PaddingSize"->1],
				NormalizationLayer[],Ramp,ConvolutionLayer[c,{3,3},"PaddingSize"->1]},
				"plus"->TotalLayer[],
				"short"->{ResizeLayer[size],ConvolutionLayer[c,{1,1}]}|>,
				{NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]


contract[channel_,crop_:{{1,1},{1,1}}]:=NetGraph[{"conv"->res[channel],"pooling"->PoolingLayer[2,2,"Function"->Mean],
										"cropping"->PartLayer[{;;,crop[[1,1]];;-crop[[1,-1]],crop[[2,1]];;-crop[[2,-1]]}]},
										{NetPort["Input"]->"conv"->"pooling"->NetPort["Pooling"],"conv"->"cropping"->NetPort["Shortcut"]}];


expand[channel_,size_]:=NetGraph[{"deconv"->upres[channel,size],
						"join"->CatenateLayer[],
						"conv"->res[channel/2]},
						{NetPort["Input"]->"deconv"->"join",
						NetPort["Shortcut"]->"join"->"conv"}];


UNet=NetInitialize[NetGraph[<|Table[{"contract_"<>ToString[i]->contract[c*2^(i-1)],
					"expand_"<>ToString[i]->expand[c*2^Max[(i-1),1],size/2^(i-1)],
					"\[Lambda]_F_"<>ToString[i]->{LinearLayer[c*2^(i-1)],ReplicateLayer[Floor[size/2^i],2]},
					"\[Lambda]_B_"<>ToString[i]->{LinearLayer[c*2^(i-1)],ReplicateLayer[Floor[size/2^i],2]},
					"thread_F_"<>ToString[i]->ThreadingLayer[Plus],
					"thread_B_"<>ToString[i]->ThreadingLayer[Plus]},{i,depth}],
					"\[Lambda]_F_0"->{LinearLayer[c],ReplicateLayer[size,2]},"thread_F_0"->ThreadingLayer[Plus],
					"\[Lambda]_B_0"->{LinearLayer[c],ReplicateLayer[size,2]},"thread_B_0"->ThreadingLayer[Plus],
					"preprocess"->ConvolutionLayer[c,{1,1}],
					"postprocess"->{ConvolutionLayer[c,{1,1}],NormalizationLayer[],ElementwiseLayer["GELU"],ConvolutionLayer[channel,{1,1}]},
					"ubase"->res[c*2^(depth-1)],
					"\[Lambda]Process"->{LinearLayer[2\[Lambda]latent],ElementwiseLayer["GELU"],LinearLayer[3\[Lambda]latent],ElementwiseLayer["GELU"]},
					"Loss"->MeanSquaredLossLayer[],
					"Rescale"->ThreadingLayer[Times]|>,
 
 Flatten[{
		 NetPort["NoiseT"]->"preprocess"->"thread_F_0"->"contract_1","preprocess"->"thread_B_0",
		 NetPort["\[Lambda]"]->"\[Lambda]Process"->"\[Lambda]_F_0"->"thread_F_0","\[Lambda]Process"->"\[Lambda]_B_0"->"thread_B_0",
		 Table[{NetPort["contract_"<>ToString[i],"Pooling"]->"thread_F_"<>ToString[i]->
		 If[i<depth,"contract_"<>ToString[i+1],"ubase"->"thread_B_"<>ToString[depth]->NetPort["expand_"<>ToString[depth],"Input"]],
		 "\[Lambda]Process"->"\[Lambda]_F_"<>ToString[i]->"thread_F_"<>ToString[i],
		 "\[Lambda]Process"->"\[Lambda]_B_"<>ToString[i]->"thread_B_"<>ToString[i],
		 NetPort["contract_"<>ToString[i],"Shortcut"]->NetPort["expand_"<>ToString[i],"Shortcut"],
		 NetPort["expand_"<>ToString[i],"Output"]->"thread_B_"<>ToString[i-1]->If[i>1,NetPort["expand_"<>ToString[i-1],"Input"],"postprocess"]},{i,depth}],
		 "postprocess"->"Loss",
		 NetPort["Noise"]->"Loss"->"Rescale",
		 NetPort["Scale"]->"Rescale"->NetPort["Loss"]}],
		 "NoiseT"->Prepend[size,channel],
		 "\[Lambda]"->\[Lambda]latent]];
 
 


length=Round[Length[ndata]*0.9];

validation=Block[{data=ndata[[length+1;;]],u,\[Lambda]s,scale,noise,\[Lambda]embedding,nT},
	u=RandomReal[UniformDistribution[{0,1}],Length[data]];
	\[Lambda]s=\[Lambda][u];
	scale=(\[Sigma][\[Lambda][u]])^2;
	noise=RandomReal[NormalDistribution[0,1],Join[{Length[data],channel},size]];
	\[Lambda]embedding=Map[\[Lambda]Encoding,\[Lambda]s];
	nT=Table[Fzx[\[Lambda]s[[i]],{data[[i]]},noise[[i]]],{i,Length[data]}];
 
Table[ <|"NoiseT"->nT[[i]],
			"Noise"->noise[[i]],
			"\[Lambda]"->\[Lambda]embedding[[i]],
			"Scale"->scale[[i]]|>,{i,Length[nT]}]];

GlobeLoss=Infinity;

Report[net_,loss_]:=Block[{},
				Print[{GlobeLoss,loss}];
				If[loss<GlobeLoss,Block[{},Print["Update"];
				Set[GlobeLoss,loss];
				Export["/data/DDPM_UnCondition_trained_"<>name<>".mx",net]]]];



(* ::Subsection:: *)
(*Training*)


trained=NetTrain[UNet,
				{Function[Block[{u,\[Lambda]s,noise,select,\[Lambda]embedding,scale},
							u=RandomReal[UniformDistribution[{0,1}],batch];
							\[Lambda]s=\[Lambda][u];
							scale=(\[Sigma][\[Lambda][u]])^2;
							noise=RandomReal[NormalDistribution[0,1],Join[{batch,channel},size]];
							select=RandomSample[Range[length],batch];
							\[Lambda]embedding=Map[\[Lambda]Encoding,\[Lambda]s];
							t=ndata[[select]];
							nT=Table[Fzx[\[Lambda]s[[i]],{t[[i]]},noise[[i]]],{i,batch}];
									<|"NoiseT"->nT,
									"Noise"->noise,
									"\[Lambda]"->\[Lambda]embedding,
									"Scale"->scale|>]],"RoundLength" ->batch*1000},
				LossFunction -> {"Loss"->Scaled[Prepend[size ,channel]/.List->Times]},
				BatchSize -> batch,
				MaxTrainingRounds -> 1000,
				ValidationSet -> validation,
				TargetDevice -> {"GPU",gpuID},
				Method -> {"ADAM","LearningRate"->rate,"L2Regularization"->10^-6},
TrainingProgressReporting -> {{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];


(* ::Subsection:: *)
(*Unconditional sample*)


net=NetTake[Import["/data/DDPM_UnCondition_trained_"<>name<>".mx"],"postprocess"];  (*Pre-trained model*)

Report[net_]:=Block[{batch=64},
			\[Lambda]seq=Table[\[Lambda][i],{i,1,0.00,-1*10^-3}];
			\[Sigma]seq[v_]:=Sqrt[(1-Exp[\[Lambda]seq[[1;;-2]]-\[Lambda]seq[[2;;]]])](\[Sigma][\[Lambda]seq[[1;;-2]]])^(1-v)*\[Sigma][\[Lambda]seq[[2;;-1]]]^v;
			\[Alpha]seq=Map[\[Alpha],\[Lambda]seq];
			initial=RandomReal[NormalDistribution[0,1],Join[{batch,channel},size]];
			
			Table[Set[initial,
						initial*\[Alpha]seq[[t+1]]/\[Alpha]seq[[t]]+\[Alpha]seq[[t+1]]*(Sqrt[(1-\[Alpha]seq[[t+1]]^2)/(\[Alpha]seq[[t+1]]^2)]-Sqrt[(1-\[Alpha]seq[[t]]^2)/(\[Alpha]seq[[t]]^2)])*
						Normal[net[<|"NoiseT"->initial,
						"\[Lambda]"->Table[\[Lambda]Encoding[\[Lambda]seq[[t]]],batch],
						"Condition"->condition|>,TargetDevice->"GPU"]]];
				Print[{t,Map[MinMax,initial[[1;;-1;;200]]]}];
				initial,{t,Length[\[Lambda]seq]-1}][[-1]]];

{mean,sigma} = {14.6,12.7};
result = (Report[net])*sigma+mean;
Export["/data/result/sample.mx",result]; (*unconditional result*)


(* ::Subsection:: *)
(*Observation*)


obser = Import["/data/observation2021.mx"];
aa = Array[0&,{120,120}];
ppp = Position[obser[[1]],_?(#!=-99&)];
tttt = Table[Set[aa[[ppp[[i,1]],ppp[[i,2]]]],1],{i,1,120}];
aaa = ArrayReshape[aa,{1,1,120,120}];
e = Array[1&,{1,1,120,120}];


(* ::Subsection:: *)
(*CLIN*)


Table[
nn = i;
nnn = ToString[nn];
nnobser = (obser[[nn]]*aa-14.6)/12.7;
yt = ArrayReshape[nnobser,{1,1,120,120}];

Report[net_]:=Block[{batch=1},
			\[Lambda]seq=Table[\[Lambda][i],{i,1,0,-1*10^-3}];
			\[Sigma]seq[v_]:=Sqrt[(1-Exp[\[Lambda]seq[[1;;-2]]-\[Lambda]seq[[2;;]]])](\[Sigma][\[Lambda]seq[[1;;-2]]])^(1-v)*\[Sigma][\[Lambda]seq[[2;;-1]]]^v;
			\[Alpha]seq=Map[\[Alpha],\[Lambda]seq];
			
			initial = RandomReal[NormalDistribution[0,1],Join[{batch,1},size]];
			
			Table[Set[initial,
						ArrayReshape[Fzx[\[Lambda][(0.01/t)],ArrayReshape[initial*(e-aaa)+yt*aaa,{120,120}],
						RandomReal[NormalDistribution[0,1],{120,120}]],{1,1,120,120}]*\[Alpha]seq[[t+1]]/\[Alpha]seq[[t]]+\[Alpha]seq[[t+1]]*(Sqrt[(1-\[Alpha]seq[[t+1]]^2)/(\[Alpha]seq[[t+1]]^2)]-Sqrt[(1-\[Alpha]seq[[t]]^2)/(\[Alpha]seq[[t]]^2)])*
						Normal[net[<|"NoiseT"->initial,
						"\[Lambda]"->Table[\[Lambda]Encoding[\[Lambda]seq[[t]]],batch],
						"Condition"->condition|>,TargetDevice->{"GPU",gpuID}]]];
				initial,{t,Length[\[Lambda]seq]-1}][[-1]]];
			
				
{mean,sigma} = {14.6,12.7};
result = (Report[net])*sigma+mean;
Export["/data/result/"<>nnn<>".mx",result],{i,2920}]; (*CLIN result*)
