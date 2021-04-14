import pytest
import torch
import tormentor
import json
import tempfile
import os

# annotations from validation 2017 samples [1396, 1425, 1429, 1447, 1495]
data_str = r"""
[[[3,427,640],[{"segmentation":[[5.51,367.31,60.61,359.97,170.8,356.29,178.15,353.54,184.57,286.5,235.08,223.14,
291.09,190.08,356.29,191.0,399.45,216.71,460.98,178.15,498.63,154.27,505.05,160.7,505.05,174.47,511.48,175.39,517.91,
202.94,539.03,225.9,542.7,250.69,537.19,271.81,502.3,279.16,490.36,286.5,478.42,324.15,463.73,348.95,421.49,362.72,
444.45,368.23,450.88,377.41,424.25,393.94,394.86,391.19,382.92,379.25,356.29,382.0,351.7,361.8,341.6,374.66,352.62,
385.68,297.52,397.62,185.49,394.86,98.26,389.35,28.47,393.02,7.35,398.53]],"area":64590.187750000005,"iscrowd":0,
"image_id":163155,"bbox":[5.51,154.27,537.19,244.26],"category_id":17,"id":46853}]],[[3,427,640],[{"segmentation":
[[252.9,273.61,290.47,276.36,298.72,263.53,306.05,253.45,302.38,228.71,290.47,215.88,291.39,186.56,286.8,179.23,
276.73,179.23,271.23,181.06,268.48,192.06,259.32,208.55,269.39,211.3,250.15,216.8,233.66,236.96,233.66,252.53,243.74,
274.53,243.74,284.61,246.49,291.02,251.98,303.85,264.81,301.1,250.15,282.77],[285.89,276.36,276.73,303.85,258.4,
312.09,273.06,312.09,279.47,306.6,291.39,312.09,284.06,304.76]],"area":5038.048199999999,"iscrowd":0,"image_id":
166259,"bbox":[233.66,179.23,72.39,132.86],"category_id":16,"id":36967},{"segmentation":[[309.96,249.71,320.94,233.25,
332.82,224.1,349.28,220.45,356.6,222.27,363.0,199.42,358.42,173.81,365.74,160.1,370.31,148.21,384.94,149.13,391.34,
170.16,389.51,182.96,392.25,193.93,382.2,192.1,379.45,212.22,388.6,235.99,405.97,247.88,405.97,260.68,404.14,277.14,
391.34,290.85,365.74,306.39,374.88,328.34,366.65,324.68,356.6,326.51,355.68,324.68,349.28,325.6,341.05,322.85,339.22,
313.71,347.45,306.39,331.91,304.57,338.31,323.77,351.11,332.0,369.4,332.0,374.88,337.48,348.15,336.22,340.32,336.93,
337.47,326.96,324.64,304.87,308.97,296.32,303.98,290.62,294.72,293.47,303.98,285.64,299.71,279.22,299.71,268.54,
301.13,260.7]],"area":9740.558750000002,"iscrowd":0,"image_id":166259,"bbox":[294.72,148.21,111.25,189.27],
"category_id":16,"id":41598},{"segmentation":[[167.13,247.94,164.37,239.67,155.19,232.32,167.13,220.39,172.64,217.63,
179.98,205.69,189.17,198.35,192.84,192.84,192.84,179.98,195.59,168.05,205.69,165.29,206.61,168.05,211.2,171.72,213.04,
186.41,213.04,198.35,215.8,207.53,220.39,210.29,227.73,217.63,231.41,232.32,234.16,243.34,226.82,252.53,213.96,259.87,
213.04,270.89,213.04,283.75,212.12,294.77,202.94,295.69,198.35,295.69,191.92,294.77,198.35,280.08,203.86,272.73,
205.69,268.14,205.69,266.3,199.27,260.79,192.84,264.46,192.84,269.06,192.84,276.4,189.17,277.32,183.66,281.91,190.08,
291.09,190.08,297.52,185.49,299.36,181.82,295.69,179.98,283.75,173.55,263.55,171.72,259.87]],"area":5027.412799999998,
"iscrowd":0,"image_id":166259,"bbox":[155.19,165.29,78.97,134.07],"category_id":16,"id":42867},{"segmentation":
[[291.72,207.12,296.52,205.93,296.52,198.33,299.72,191.14,300.12,185.14,302.91,181.94,308.91,180.74,316.1,182.74,
320.9,189.94,320.9,197.93,323.3,203.13,321.7,207.12,326.9,211.52,333.69,218.32,336.09,224.31,316.5,237.5,309.71,
251.09,306.51,253.49,301.32,229.91,289.32,214.32,290.52,208.32,295.32,206.72],[315.1,302.05,293.04,302.56,287.4,
302.05,287.4,302.05],[288.42,302.56,291.5,293.32,289.45,288.71,288.94,280.5,288.42,278.45,292.01,287.17,294.58,295.89,
299.71,298.97,301.76,297.94,303.3,294.86,304.84,291.27,311.51,297.43,307.92,298.45,314.58,301.53]],"area":
1857.9427499999988,"iscrowd":0,"image_id":166259,"bbox":[287.4,180.74,48.69,121.82],"category_id":16,"id":43297}]],
[[3,640,587],[{"segmentation":[[164.08,96.5,153.93,113.28,152.54,125.87,161.63,170.64,170.72,221.35,171.42,232.54,
169.32,245.83,172.12,259.46,179.81,267.16,195.9,272.75,216.88,271.01,236.47,260.16,239.96,253.17,241.36,244.78,238.57,
236.03,235.77,227.29,233.32,221.0,232.62,184.27,231.22,153.5,227.38,111.88,217.93,99.64,209.54,90.9,205.52,80.98,
200.58,42.83,196.54,25.77,192.5,13.65,188.01,11.86,184.87,11.41,176.79,10.51,171.41,12.3,167.37,13.2,165.57,16.79,
166.92,49.56,167.37,76.49,167.37,86.81]],"area":14806.775500000003,"iscrowd":0,"image_id":166426,"bbox":[152.54,10.51,
88.82,262.24],"category_id":44,"id":83858},{"segmentation":[[318.93,200.48,306.72,199.09,298.34,194.55,297.64,190.37,
293.11,95.47,292.41,68.6,296.25,64.07,300.44,59.53,302.53,51.86,302.18,26.39,300.78,1.62,324.86,1.62,324.86,38.95,
325.21,56.39,327.65,61.28,334.28,66.16,336.72,124.77,337.42,168.73,329.74,171.87,323.81,175.01,318.23,179.2,315.79,
184.43,317.53,195.25]],"area":6535.100750000003,"iscrowd":0,"image_id":166426,"bbox":[292.41,1.62,45.01,198.86],
"category_id":44,"id":86415},{"segmentation":[[213.39,50.4,223.37,39.51,258.79,27.7,295.11,24.98,310.55,24.07,326.89,
21.34,335.06,24.07,354.13,31.33,359.58,33.15,371.39,40.41,343.24,42.23,290.57,40.41,283.31,41.32,275.13,44.05,271.5,
45.86,256.97,51.31,229.73,56.76,209.75,53.13]],"area":2566.649049999999,"iscrowd":0,"image_id":166426,"bbox":[209.75,
21.34,161.64,35.42],"category_id":62,"id":106025},{"segmentation":[[1.51,108.42,0.0,632.47,587.0,630.96,587.0,75.29,
387.01,42.16,254.49,39.15,150.59,54.21,28.61,85.84,1.51,105.41]],"area":337142.1807,"iscrowd":0,"image_id":166426,
"bbox":[0.0,39.15,587.0,593.32],"category_id":67,"id":121303},{"segmentation":[[241.28,75.29,237.86,135.19,232.73,
138.61,229.3,111.23,215.61,95.83,207.06,83.85,207.06,75.29]],"area":1099.7273,"iscrowd":0,"image_id":166426,"bbox":
[207.06,75.29,34.22,63.32],"category_id":47,"id":683175},{"segmentation":[[23.51,506.62,79.45,416.83,81.66,410.94,
107.42,350.59,98.59,396.22,113.31,352.8,108.15,386.65,119.19,353.53,114.78,395.48,127.29,354.27,116.25,407.26,106.68,
419.77,88.28,427.13,27.93,550.05,12.47,555.2,2.17,547.1,18.36,508.83]],"area":4512.919449999999,"iscrowd":0,
"image_id":166426,"bbox":[2.17,350.59,125.12,204.61],"category_id":48,"id":690431},{"segmentation":[[521.84,352.48,
516.08,350.17,508.59,354.21,504.56,359.97,498.22,390.5,502.25,424.48,504.56,430.24,510.9,433.13,514.93,441.19,516.08,
486.12,517.23,524.14,518.96,533.93,521.84,543.73,529.9,547.76,538.55,544.3,544.88,535.09,532.78,479.79,523.57,431.4,
525.3,431.4,524.14,401.44]],"area":3827.9363999999964,"iscrowd":0,"image_id":166426,"bbox":[498.22,350.17,46.66,
197.59],"category_id":49,"id":695695},{"segmentation":[[238.71,86.94,241.5,79.95,251.6,66.02,250.51,61.92,245.05,
60.56,240.13,65.75,237.95,77.22,238.49,86.77],[239.58,87.05,241.5,90.33,238.49,97.43,238.22,87.05]],"area":
181.49865000000008,"iscrowd":0,"image_id":166426,"bbox":[237.95,60.56,13.65,36.87],"category_id":49,"id":700222},
{"segmentation":[[400.62,207.52,407.27,200.87,410.6,82.84,402.29,67.88,403.95,4.71,377.35,3.05,374.03,59.57,362.39,
67.88,362.39,164.29,385.66,165.96,405.61,177.59]],"area":6623.8863500000025,"iscrowd":0,"image_id":166426,"bbox":
[362.39,3.05,48.21,204.47],"category_id":44,"id":1865428},{"segmentation":[[338.22,272.05,357.73,278.55,381.69,274.79,
391.62,262.81,402.91,185.41,398.8,177.54,385.11,171.03,371.76,167.95,351.23,167.61,336.17,170.69,328.59,173.13,319.01,
180.32,316.95,187.17,328.74,250.85,331.94,263.11]],"area":7523.312600000004,"iscrowd":0,"image_id":166426,"bbox":
[316.95,167.61,85.96,110.94],"category_id":47,"id":1880775},{"segmentation":[[452.2,66.03,453.16,91.81,462.7,101.83,
460.79,107.56,447.43,107.56,445.04,100.4,450.29,88.47,440.27,66.98]],"area":375.6713499999999,"iscrowd":0,"image_id":
166426,"bbox":[440.27,66.03,22.43,41.53],"category_id":48,"id":2102732}]],[[3,425,640],[{"segmentation":[[501.94,
249.44,533.6,254.03,563.73,256.59,575.47,249.44,593.35,253.52,582.62,275.99,565.26,272.42,558.11,267.82,492.24,
263.23]],"area":1406.7813000000015,"iscrowd":0,"image_id":167572,"bbox":[492.24,249.44,101.11,26.55],"category_id":48,
"id":686586},{"segmentation":[[473.05,295.23,538.15,293.1,540.81,297.37,600.57,289.9,597.37,297.37,583.5,304.84,
531.21,314.97,523.74,316.58,522.14,314.97,450.11,322.98,447.44,321.38,466.12,304.84]],"area":2609.7283500000012,
"iscrowd":0,"image_id":167572,"bbox":[447.44,289.9,153.13,33.08],"category_id":49,"id":697794},{"segmentation":
[[129.15,290.93,101.42,217.21,130.84,132.75,150.9,98.37,186.24,72.58,204.38,54.44,212.02,51.57,223.48,50.62,233.03,
57.3,238.76,67.81,251.18,77.36,268.37,108.88,279.83,209.16,256.91,314.21,227.3,347.64,204.38,351.46,168.09,348.6,
144.21,325.67,127.98,291.29]],"area":38445.28489999999,"iscrowd":0,"image_id":167572,"bbox":[101.42,50.62,178.41,
300.84],"category_id":51,"id":1533850},{"segmentation":[[307.53,379.16,312.3,353.37,321.85,290.34,338.09,233.03,340.0,
205.34,347.64,182.42,374.38,173.82,405.9,185.28,442.19,192.92,474.66,202.47,495.67,208.2,505.22,220.62,509.04,237.81,
485.17,274.1,470.84,290.34,439.33,326.63,395.39,359.1,374.38,377.25,356.24,389.66,337.13,388.71,321.85,379.16]],
"area":26452.666849999998,"iscrowd":0,"image_id":167572,"bbox":[307.53,173.82,201.51,215.84],"category_id":54,"id":
1552549},{"segmentation":[[324.72,158.54,398.26,189.1,467.98,203.43,509.04,199.61,518.6,172.87,495.67,132.75,460.34,
85.96,407.81,42.98,355.28,13.37,337.13,2.87,318.03,9.55,309.44,67.81,302.75,114.61,312.3,152.81,328.54,168.09]],
"area":26337.427,"iscrowd":0,"image_id":167572,"bbox":[302.75,2.87,215.85,200.56],"category_id":54,"id":1552995},
{"segmentation":[[207.25,295.11,186.24,390.62,170.96,398.26,154.72,382.02,144.21,360.06,145.17,318.03,148.99,297.02,
164.27,279.83,186.24,273.15,186.24,282.7,192.92,283.65,195.79,279.83]],"area":5489.683850000001,"iscrowd":0,
"image_id":167572,"bbox":[144.21,273.15,63.04,125.11],"category_id":55,"id":1556559},{"segmentation":[[154.72,67.81,
95.51,8.6,95.51,0.0,0.96,0.0,0.96,114.61,22.92,126.07,50.62,131.8,54.44,129.89,48.71,122.25,13.37,106.01,34.38,88.82,
77.36,116.52,87.87,115.56,84.04,103.15,42.98,79.27,86.91,16.24,151.85,72.58]],"area":8394.576549999998,"iscrowd":0,
"image_id":167572,"bbox":[0.96,0.0,153.76,131.8],"category_id":62,"id":1586097},{"segmentation":[[483.26,126.07,
605.51,138.48,612.19,151.85,630.34,200.56,593.09,399.21,586.4,418.31,521.46,416.4,527.19,403.03,498.54,371.52,507.13,
314.21,529.1,314.21,567.3,308.48,602.64,297.98,604.55,288.43,504.27,297.02,510.0,264.55,562.53,270.28,573.99,274.1,
593.09,272.19,595.96,250.22,578.76,251.18,565.39,258.82,510.0,247.36,507.13,170.0,503.31,148.99,486.12,130.84],
[127.98,139.44,71.63,149.94,65.9,257.87,73.54,323.76,83.09,351.46,87.87,366.74,96.46,387.75,99.33,402.08,113.65,416.4,
153.76,420.22,170.96,398.26,147.08,392.53,122.25,281.74,107.92,235.9,114.61,170.96,122.25,150.9,127.02,140.39]],
"area":40522.653300000005,"iscrowd":0,"image_id":167572,"bbox":[65.9,126.07,564.44,294.15],"category_id":67,"id":
1617311},{"segmentation":[[514.78,129.89,509.04,97.42,542.47,37.25,559.66,10.51,584.49,0.0,640.0,0.0,640.0,149.94,
612.19,150.9,586.4,132.75]],"area":15106.377149999997,"iscrowd":0,"image_id":167572,"bbox":[509.04,0.0,130.96,150.9],
"category_id":1,"id":1729952},{"segmentation":[[172.72,1.17,191.56,37.81,211.45,53.51,231.34,59.79,242.86,60.84,
259.61,55.6,270.07,38.85,282.64,35.71,295.2,37.81,307.76,31.53,316.13,25.25,323.46,12.68,321.37,1.17,322.8,0.0],
[341.64,0.0,374.09,25.65,397.12,37.17,414.92,48.68,434.81,61.24,448.41,77.99,455.74,92.65,469.35,112.54,482.96,119.86,
502.85,128.24,509.13,127.19,502.85,84.27,499.71,59.15,502.85,40.31,503.89,25.65,505.99,13.09,514.36,0.0]],"area":
16419.9829,"iscrowd":0,"image_id":167572,"bbox":[172.72,0.0,341.64,128.24],"category_id":1,"id":1754718},
{"segmentation":[[269.97,425.0,272.18,415.02,277.24,404.9,290.2,396.37,309.17,391.94,325.62,393.21,340.48,397.0,
352.18,410.28,358.18,422.29,358.18,425.0,271.55,425.0]],"area":2267.863300000001,"iscrowd":0,"image_id":167572,"bbox":
[269.97,391.94,88.21,33.06],"category_id":47,"id":2100665},{"segmentation":[[153.57,422.49,184.08,388.0,192.04,381.37,
199.12,374.74,203.98,369.87,207.52,364.57,211.94,359.7,215.92,357.49,256.15,372.53,262.78,373.85,265.88,377.39,268.09,
383.58,270.3,396.4,269.42,407.01,267.21,415.41,265.44,424.26,152.25,425.0]],"area":5244.182149999999,"iscrowd":0,
"image_id":167572,"bbox":[152.25,357.49,118.05,67.51],"category_id":47,"id":2101315},{"segmentation":[[564.98,0.01,
535.12,44.44,520.56,68.47,501.62,80.85,503.8,88.87,514.0,89.59,510.36,124.55,527.84,128.92,524.2,99.06,519.83,85.22,
532.94,70.66,529.3,61.19,543.13,49.54,577.36,2.2]],"area":1622.8492500000018,"iscrowd":0,"image_id":167572,"bbox":
[501.62,0.01,75.74,128.91],"category_id":62,"id":2120889},{"segmentation":[[81.65,120.5,46.06,100.61,32.45,95.38,15.7,
82.82,4.19,82.82,0.0,4.31,97.35,1.17,154.93,63.98,153.88,72.35,98.4,31.53,50.25,76.54,56.53,90.15,88.98,112.13]],
"area":8356.3565,"iscrowd":0,"image_id":167572,"bbox":[0.0,1.17,154.93,119.33],"category_id":67,"id":2225955}]],[[3,
480,640],[{"segmentation":[[139.75,146.24,130.62,116.56,116.54,70.91,108.94,56.07,97.14,35.14,97.9,27.05,96.38,15.26,
92.2,0.0,23.33,0.42,35.89,29.34,43.12,40.75,40.83,53.3,45.78,60.15,50.73,68.14,53.39,69.66,54.53,80.7,58.72,89.45,
59.48,97.06,61.38,105.81,64.04,116.84,65.94,128.64,72.41,140.15,77.74,146.62,78.5,153.85,81.16,162.98,83.06,169.06,
94.86,170.21,114.64,167.16,126.06,160.31,134.05,154.23]],"area":10560.089350000002,"iscrowd":0,"image_id":173004,
"bbox":[23.33,0.0,116.42,170.21],"category_id":44,"id":80525},{"segmentation":[[92.04,0.05,99.29,30.41,97.48,37.67,
117.42,76.19,127.84,113.81,138.27,121.06,143.7,121.06,147.78,117.89,127.84,22.26,132.83,16.37,153.22,12.29,151.41,
1.41,93.4,0.5]],"area":3466.6531499999996,"iscrowd":0,"image_id":173004,"bbox":[92.04,0.05,61.18,121.01],
"category_id":44,"id":81340},{"segmentation":[[575.14,51.89,527.57,51.89,514.59,58.38,495.14,58.38,488.65,51.89,
397.84,48.65,407.57,64.86,516.76,75.68,502.7,83.24,418.38,72.43,415.14,107.03,375.14,123.24,250.81,120.0,210.81,
100.54,215.14,70.27,220.54,57.3,192.43,58.38,169.73,57.3,169.73,97.3,141.62,116.76,134.05,118.92,138.38,145.95,132.97,
160.0,88.65,172.97,82.16,190.27,83.24,220.54,52.97,242.16,23.78,234.59,7.57,211.89,0.0,211.89,2.16,480.0,640.0,476.76,
638.92,264.86,575.14,252.97,487.57,218.38,447.57,182.7,419.46,153.51,421.62,131.89,446.49,99.46,467.03,90.81,496.22,
88.65,541.62,84.32,628.11,89.73]],"area":204014.3883,"iscrowd":0,"image_id":173004,"bbox":[0.0,48.65,640.0,431.35],
"category_id":67,"id":390301},{"segmentation":[[124.84,24.7,139.91,80.66,156.05,148.47,171.12,152.77,190.49,147.39,
214.17,140.93,192.65,13.94,166.82,6.4,147.44,7.48,127.0,13.94]],"area":8909.335400000002,"iscrowd":0,"image_id":
173004,"bbox":[124.84,6.4,89.33,146.37],"category_id":47,"id":677470},{"segmentation":[[398.21,222.55,416.45,235.95,
433.55,242.51,442.67,246.21,459.78,248.49,482.29,254.19,557.83,286.12,581.2,293.53,583.48,296.09,580.35,302.36,576.07,
304.93,569.8,304.64,565.24,303.79,509.09,275.57,470.32,258.18,461.49,255.33,446.09,252.48,435.26,252.48,428.99,254.48,
418.16,256.76,410.75,253.62,412.46,246.87,415.03,246.02,417.31,245.45,413.32,242.88,409.61,238.89,417.59,242.88,
419.87,242.6,406.19,233.76]],"area":1912.946350000004,"iscrowd":0,"image_id":173004,"bbox":[398.21,222.55,185.27,
82.38],"category_id":48,"id":686273},{"segmentation":[[213.11,57.78,263.81,62.34,272.46,66.91,281.83,73.4,298.65,
71.95,316.19,67.87,318.11,65.47,302.25,68.59,316.43,63.79,300.33,67.63,318.35,61.14,318.11,60.18,296.25,65.71,284.71,
66.19,270.3,60.42,251.55,57.3,197.01,49.13,197.01,56.58]],"area":666.5897500000004,"iscrowd":0,"image_id":173004,
"bbox":[197.01,49.13,121.34,24.27],"category_id":48,"id":687964},{"segmentation":[[145.59,365.77,231.51,278.44,272.82,
246.51,282.68,230.08,277.05,216.93,246.06,245.57,217.89,271.87,169.53,315.53,157.79,323.04,123.05,354.97,129.62,
368.59,139.01,368.12]],"area":3676.2540499999977,"iscrowd":0,"image_id":173004,"bbox":[123.05,216.93,159.63,151.66],
"category_id":49,"id":694993},{"segmentation":[[295.55,182.17,263.19,284.64,363.51,307.29,414.2,245.81,404.49,229.63,
362.43,198.35]],"area":12123.273100000002,"iscrowd":0,"image_id":173004,"bbox":[263.19,182.17,151.01,125.12],
"category_id":59,"id":1072281},{"segmentation":[[569.57,161.31,507.65,167.0,456.41,164.86,452.14,146.36,471.36,120.03,
512.63,102.24,547.5,108.64,561.03,127.15,578.82,151.34,585.93,167.0,572.41,159.88]],"area":5969.755350000001,
"iscrowd":0,"image_id":173004,"bbox":[452.14,102.24,133.79,64.76],"category_id":59,"id":1074015},{"segmentation":
[[319.59,117.18,313.56,109.21,303.9,101.97,295.22,93.53,286.29,88.7,284.11,84.6,289.66,83.63,300.04,78.32,310.9,72.05,
319.11,63.84,324.18,57.56,331.42,58.29,337.69,62.63,350.0,60.46,358.93,60.7,363.76,59.98,370.04,64.56,376.55,74.7,
378.72,82.42,377.52,86.77,372.44,97.71,360.39,106.75,339.3,116.39,328.45,120.61,319.41,118.8]],"area":3662.41605,
"iscrowd":0,"image_id":173004,"bbox":[284.11,57.56,94.61,63.05],"category_id":59,"id":1074488},{"segmentation":
[[568.84,164.33,515.48,168.97,469.08,185.98,483.77,202.22,511.61,221.56,580.44,246.3,620.66,253.26,639.99,254.04,
640.0,253.26,640.0,205.32,605.19,178.25,571.94,162.78]],"area":9823.527000000004,"iscrowd":0,"image_id":173004,"bbox":
[469.08,162.78,170.92,91.26],"category_id":59,"id":1075081},{"segmentation":[[215.73,55.01,203.87,1.08,337.62,1.08,
338.7,42.07,310.65,37.75,313.89,10.79,264.27,12.94,271.82,42.07,258.88,40.99,255.64,8.63,239.46,8.63,240.54,50.7,
232.99,49.62,229.75,4.31,220.04,5.39,225.44,56.09]],"area":3198.722350000001,"iscrowd":0,"image_id":173004,"bbox":
[203.87,1.08,134.83,55.01],"category_id":62,"id":1586423},{"segmentation":[[534.5,16.39,528.67,50.99,520.77,50.81,
526.98,10.94,530.74,0.6,539.2,0.0],[525.28,0.41,514.0,4.17,499.9,5.11,492.94,50.43,486.73,50.62,492.0,10.37,492.56,
4.73,454.95,3.04,447.99,0.03,500.84,0.22],[433.93,50.68,438.27,0.19,430.21,0.81,421.85,52.85,433.93,52.54],[371.61,
43.49,362.83,42.47,365.28,13.04,366.91,0.98,376.93,0.37]],"area":1923.3809999999994,"iscrowd":0,"image_id":173004,
"bbox":[362.83,0.0,176.37,52.85],"category_id":62,"id":1587272},{"segmentation":[[363.3,59.49,444.62,66.74,499.47,
75.56,501.36,78.4,501.05,79.98,425.08,70.52,383.79,66.74,365.5,62.64]],"area":711.9657000000001,"iscrowd":0,
"image_id":173004,"bbox":[363.3,59.49,138.06,20.49],"category_id":49,"id":1892423},{"segmentation":[[4.96,46.01,42.17,
54.69,54.57,72.06,62.02,95.62,69.46,137.79,81.86,152.68,84.34,178.72,76.9,197.33,84.34,203.53,86.82,225.86,49.61,
239.5,24.81,238.26,13.64,207.25,2.48,202.29,0.0,46.01]],"area":12446.482250000001,"iscrowd":0,"image_id":173004,
"bbox":[0.0,46.01,86.82,193.49],"category_id":44,"id":2095277}]]]"""

# dataset is like a torchvsion Coco Dataset with 5 samples.
dataset = [[torch.rand(size=datum[0]),datum[1]] for datum in json.loads(data_str)]
testable_augmentations = [tormentor.RandomWrap, tormentor.RandomBrightness, tormentor.RandomIdentity]

@pytest.mark.parametrize("augmentation_factory", testable_augmentations)
def test_show_augmentation(augmentation_factory):
    fd, tempfile_path = tempfile.mkstemp(suffix=".png", prefix=None, dir=None, text=False)
    os.close(fd)
    os.remove(tempfile_path)
    for mask in [True, False]:
        aug_ds = tormentor.AugmentedCocoDs(dataset, augmentation_factory, add_mask=mask)
        for device in ["cpu", "cuda"]:
            for n in range(len(aug_ds)):
                aug_ds.show_augmentation(n,save_filename=tempfile_path, device=device)
                assert os.path.exists(tempfile_path)
                os.remove(tempfile_path)


@pytest.mark.parametrize("augmentation_factory", testable_augmentations)
def test_augment(augmentation_factory):
    for mask in [True, False]:
        for device in [torch.device("cpu"), torch.device("cuda")]:
            aug_ds = tormentor.AugmentedCocoDs(dataset, augmentation_factory, add_mask=mask, output_device=device)
            for n in range(len(aug_ds)):
                data = dataset[n]
                aug_data = aug_ds[n]
                assert len(aug_data) == len(data) + mask
                assert aug_data[0].device.type == device.type
                assert data[0].size() == aug_data[0].size() and len(data[1]) == len(data[1])
