# SRMCL_MER_main
Boosting Micro-expression Recognition via
Self-expression Reconstruction and Memory
Contrastive Learning

This project is the implementation for our paper “Boosting Micro-expression Recognition via Self-expression Reconstruction and Memory Contrastive Learning”.
The codes need to run in the environment: Python 3.8.
## Model framework 

The proposed MER framework. It includes four components, e.g., 1) the ME preprocessing module estimates optical
flow between onset and apex frame using TV-L1 for discriminative feature learning, 2) The self-expression reconstruction
module includes an encoder-decoder structure that reconstructs input ME from patch-wise masked faces, 3) the prototype-
based memory contrastive learning module includes a dynamically updated memory dictionary that stores class prototypes for
contrastive learning, and 4) the classification head predicts the ME category.

 <img src="./SRMCL.png" width = "800" height = "600" alt="SRMCL" align=center />

## Experimental Results  

The following shows the results of performance comparison on a single dataset:

<table class=MsoNormalTable border=1 cellspacing=0 cellpadding=0 width=582
 style='width:436.2pt;border-collapse:collapse;border:none;mso-border-top-alt:
 solid windowtext 1.5pt;mso-border-bottom-alt:solid windowtext 1.5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:19.85pt'>
  <td width=145 rowspan=2 style='width:109.05pt;border-top:solid windowtext 1.5pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:宋体'>Method<o:p></o:p></span></p>
  </td>
  <td width=145 colspan=2 style='width:109.05pt;border-top:solid windowtext 1.5pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>SMIC<o:p></o:p></span></p>
  </td>
  <td width=145 colspan=2 style='width:109.05pt;border-top:solid windowtext 1.5pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>CASME II<o:p></o:p></span></p>
  </td>
  <td width=145 colspan=2 style='width:109.05pt;border-top:solid windowtext 1.5pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>SAMM<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:19.85pt'>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>ACC<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>F1<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>ACC<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>F1<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>ACC</span><span lang=EN-US
  style='font-size:10.0pt;font-family:宋体'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>F1<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>LBP-TOP</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159059254
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[10]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350039003200350034000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4338<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.3421<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0;tab-stops:29.6pt'><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'>0.3968<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.3589<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.3968<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.3589<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>LBP-SIP</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159059216
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[6]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350039003200310036000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4451<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4492<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4656<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4480<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>MDMO</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159059359
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[11]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350039003300350039000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.5897<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0;tab-stops:25.05pt'><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'>0.5845<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.5169<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4966<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>Bi-WOOF</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159075049
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[12]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000370035003000340039000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6220<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0;tab-stops:25.05pt'><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'>0.6200<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.5885<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6100<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>Graph-TCN</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159075058
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[73]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000370035003000350038000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7398<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7246<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><b><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7500<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6985<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span class=SpellE><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'>LGCcon</span></span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159057004
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[18]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350037003000300034000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6502<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6400<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4090<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.3400<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span class=SpellE><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'>AUGCN+AUFusion</span></span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159057010
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[19]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350037003000310030000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7427<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7047<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7426<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><b><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7045<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>DSSN</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159058744
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[28]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350038003700340034000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6341<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6462<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7078<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7297<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.5735<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;mso-border-top-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.4644<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>KFC</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159058757
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[30]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350038003700350037000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6585<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6638<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7276<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7375<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6324<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.5709<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span class=SpellE><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'>FeatRef</span></span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159058766
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[31]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350038003700360036000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.5790<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6285<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>SLSTT</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159063543
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[41]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000360033003500340033000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7371<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7240<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7581<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7530<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7239<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6400<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span class=SpellE><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'>ExpMultNet</span></span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159058752
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[40]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000350038003700350032000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><b><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7999<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><u><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7812<o:p></o:p></span></u></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><u><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.8150<o:p></o:p></span></u></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><u><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.8009<o:p></o:p></span></u></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:14;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>MER-<span class=SpellE>Supcon</span></span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159063590
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[46]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000360033003500390030000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7358<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7286<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6765<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6251<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:15;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>I3D+MOCO</span><!--[if supportFields]><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-begin;mso-field-lock:yes'></span> REF _Ref159063634
  \r \h<span style='mso-spacerun:yes'>&nbsp; </span>\* MERGEFORMAT <span
  style='mso-element:field-separator'></span></span><![endif]--><span
  lang=EN-US style='font-size:10.0pt;font-family:"Times New Roman",serif'><sup>[48]</sup><!--[if gte mso 9]><xml>
   <w:data>08D0C9EA79F9BACE118C8200AA004BA90B02000000080000000E0000005F005200650066003100350039003000360033003600330034000000</w:data>
  </xml><![endif]--></span><!--[if supportFields]><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><span
  style='mso-element:field-end'></span></span><![endif]--><span lang=EN-US
  style='font-size:10.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7561<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7492<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7630<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7366<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6838<o:p></o:p></span></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.5436<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:16;mso-yfti-lastrow:yes;height:19.85pt'>
  <td width=145 style='width:109.05pt;border:none;border-bottom:solid windowtext 1.5pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><b><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>SRMCL<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.5pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><u><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7898<o:p></o:p></span></u></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.5pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><b><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7887<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.5pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><b><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.8320<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.5pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><b><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.8286<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:54.5pt;border:none;border-bottom:solid windowtext 1.5pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><u><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.7463<o:p></o:p></span></u></p>
  </td>
  <td width=73 style='width:54.55pt;border:none;border-bottom:solid windowtext 1.5pt;
  mso-border-top-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:19.85pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm;mso-char-indent-count:0'><u><span lang=EN-US style='font-size:10.0pt;
  font-family:"Times New Roman",serif'>0.6599<o:p></o:p></span></u></p>
  </td>
 </tr>
</table>

## Data preparation:
Due to licensing restrictions, we are unable to directly disclose or use specific datasets. To ensure the compliance of our research and the accuracy of the data, we hereby provide the official link for obtaining the dataset, so that interested researchers can apply and acquire it in accordance with the official procedures.

[SMIC](http://www.cse.oulu.fi/SMICDatabase)

[CASMEII](http://fu.psych.ac.cn/CASME/casme2-en.php)

[SAMM](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)

## Pretrained models:
The pre-training model for this method is provided here:
'/best_model/pretrain.pt'

[Link](https://pan.baidu.com/s/101yj-1d6SloGiSShCrJXeg?pwd=omyx）
Password:omyx

## Testing:
Run the following codes to reproduce the recognition results provided in the paper:

（1）Validating model performance under the self-expression reconstruction task.

`python Last_casmeII_SR.py`

（2）Validating the performance of the full SRMCL model

`python Last_casmeII_test.py`

