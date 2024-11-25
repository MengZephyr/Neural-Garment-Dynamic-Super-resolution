#include "DataIO.h"
#include "mesh.h"
#include "rayTracer.h"
#include "ZmKDTree.hpp"
#include<cstdlib>
#include<cmath>
#include<omp.h>

#define PI 3.1415926

double random()
{
	return (double(rand()) / double(RAND_MAX + 1.));
}

void UVMesh_Sample_Interpolation(R_Mesh& sourceMesh, R_Mesh& targetMesh, string infoSaveName)
{
	RayIntersection myTracer;
	myTracer.addObj(&sourceMesh);

	int failedCount = 0;
	std::vector<int> inFaceID(targetMesh.numV(), -1);
	std::vector<cv::Vec2f> inAB(targetMesh.numV(), cv::Vec2f(-1., -1.));
	std::vector<cv::Vec3f> failedVerts;
	std::vector<int> failedVInds;
	std::vector<cv::Vec3f> correctInter;

	//pass_1: some vertices at the boundary will fail to get the intersection

	for (int vi = 0; vi < targetMesh.numV(); vi++)
	{
		cv::Vec3f dir(0., 0., 1.);
		cv::Vec3f ori(targetMesh.verts[vi][0], targetMesh.verts[vi][1], -1.);
		RTCHit h = myTracer.rayIntersection(ori, dir);
		int fID = h.primID;
		double radomrr = 1.e-5;

		if (fID < 0)
		{
			if (inFaceID[vi] < 0)
			{
				printf("%d\n", vi);
				failedCount++;
				failedVerts.push_back(targetMesh.verts[vi]);
				failedVInds.push_back(vi);
			}
			else
			{
				cv::Vec3i iFace = sourceMesh.faceInds[inFaceID[vi]];
				cv::Vec3f ivert = (1. - inAB[vi][0] - inAB[vi][1]) * sourceMesh.verts[iFace[0]] + inAB[vi][0] * sourceMesh.verts[iFace[1]] + inAB[vi][1] * sourceMesh.verts[iFace[2]];
				correctInter.push_back(ivert);
			}
			continue;
		}
		else
		{
			inFaceID[vi] = fID;
			inAB[vi] = cv::Vec2f(h.u, h.v);
			cv::Vec3f interPos = myTracer.interPos(h);
			correctInter.push_back(interPos);
		} // end for else
	} // end for vi

	printf("# of failed verts = %d \n", failedCount);

	//----
	std::vector<cv::Vec3f> zerosN1(failedVerts.size(), cv::Vec3f(0., 0., 0.));
	saveObjFile(string("./test/failed.obj"), failedVerts, zerosN1);

	std::vector<cv::Vec3f> zerosN2(correctInter.size(), cv::Vec3f(0., 0., 0.));
	saveObjFile(string("./test/pass_11.obj"), correctInter, zerosN2);
	//---

	//pass_2: assume the verts must be on one triangle-face edge
	KDTree sourceMeshTree(sourceMesh.verts);
	std::vector<std::vector<int>> upVertAdjFID = sourceMesh.calcVertAdjFaceID();

	//----debug----
	/*for (int vi = 0; vi < sourceMesh.numV(); vi++)
	{
		std::vector<int>& nearFaces = upVertAdjFID[vi];
		if (nearFaces.size() > 0)
			continue;
		else
		{
			printf("id: %d \n", vi);
			cv::Vec3f vpose = sourceMesh.verts[vi];
			printf("%f, %f, %f\n", vpose[0], vpose[1], vpose[2]);
			std::vector<KDTreeLeaf> nearesetLeaves = sourceMeshTree.searchKNN(vpose, 2);
			for (int k = 0; k < 2; k++)
			{
				printf("n: %d \n", nearesetLeaves[k].id);
				printf("%f, %f, %f\n", nearesetLeaves[k].pos[0], nearesetLeaves[k].pos[1], nearesetLeaves[k].pos[2]);
			}
			
		}
			
	}*/
	//----debug----

	std::vector<cv::Vec3f> nextCorrectInter;
	for (int i = 0; i < failedVInds.size(); i++)
	{
		int vid = failedVInds[i];
		//int vid = 5931;
		//printf("%d \n", vid);
		cv::Vec3f vpose = targetMesh.verts[vid];
		KDTreeLeaf nearesetLeaf = sourceMeshTree.search(vpose);
		int UpVId = nearesetLeaf.id;
		//printf("nearestID: %d \n", UpVId);
		std::vector<int>& upFaces = upVertAdjFID[UpVId];

		if (upFaces.size() <= 0)
		{
			cv::Vec3f badpose = sourceMesh.verts[UpVId];
			std::vector<KDTreeLeaf> nearesetLeaves = sourceMeshTree.searchKNN(badpose, 2);
			UpVId = nearesetLeaves[1].id;
			upFaces = upVertAdjFID[UpVId];
		}

		float nearestDist = 1.e4;
		int fid = -1;
		cv::Vec3f abc(0., 0., 0.);

		for (int f = 0; f < upFaces.size(); f++)
		{
			int cf = upFaces[f];
			//printf("cf: %d \n", cf);
			for (int e = 0; e < 3; e++)
			{
				int i0 = sourceMesh.faceInds[cf][e % 3];
				int i1 = sourceMesh.faceInds[cf][(e + 1) % 3];
				cv::Vec3f p0 = sourceMesh.verts[i0];
				cv::Vec3f p1 = sourceMesh.verts[i1];
				cv::Vec3f p01 = p1 - p0;
				float bLen = max(norm(p01), 1.e-6);
				cv::Vec3f p0V = vpose - p0;
				float a = p0V.dot(p01) / bLen;
				if (a < 0.)
					continue;
				cv::Vec3f ccpos = (1. - a / bLen) * p0 + a / bLen * p1;
				float dist = norm(ccpos - vpose);
				if (dist < nearestDist)
				{
					abc = cv::Vec3f(0., 0., 0.);
					abc[e % 3] = 1. - a / bLen;
					abc[(e + 1) % 3] = a / bLen;
					fid = cf;
					nearestDist = dist;
				} // end for if
			} // end for e
		} // end for f

		inFaceID[vid] = fid;
		inAB[vid] = cv::Vec2f(abc[1], abc[2]);

		cv::Vec3i iFace = sourceMesh.faceInds[inFaceID[vid]];
		cv::Vec3f ivert = (1. - inAB[vid][0] - inAB[vid][1]) * sourceMesh.verts[iFace[0]] + inAB[vid][0] * sourceMesh.verts[iFace[1]] + inAB[vid][1] * sourceMesh.verts[iFace[2]];

		nextCorrectInter.push_back(ivert);

		/*printf("Debug_5931\n");
		while (1);*/
	} // end for i

	//----
	std::vector<cv::Vec3f> zerosN3(nextCorrectInter.size(), cv::Vec3f(0., 0., 0.));
	saveObjFile(string("./test/pass_2.obj"), nextCorrectInter, zerosN3);
	//----

	saveSampleInfo_InFaceAB(infoSaveName, inFaceID, inAB);
}


void Sampling_between_Different_PDResolution_Across_UV()
{
	string caseRoot = "./Data/shortskirt/Canonical/weld/";

	string sourcePD = "30";
	string targetPD = "10";

	string sourceUVName = caseRoot + "PD" + sourcePD + "_uv.ply";
	string targetUVName = caseRoot + "PD" + targetPD + "_uv.ply";

	R_Mesh sourceUVMesh;
	readPly(sourceUVName, sourceUVMesh.verts, sourceUVMesh.faceInds);
	R_Mesh targetUVMesh;
	readPly(targetUVName, targetUVMesh.verts, targetUVMesh.faceInds);

	UVMesh_Sample_Interpolation(sourceUVMesh, targetUVMesh, caseRoot + "test/" + targetPD + "_from_" + sourcePD + "_Sampling.txt");
}

void Geo_UV_Map()
{
	string caseRoot = "./Data/shortskirt/Canonical/weld/";
	string CaseName = "PD30";

	R_Mesh GeoMesh;
	readPly(caseRoot + CaseName + "_geo.ply", GeoMesh.verts, GeoMesh.faceInds);
	R_Mesh UVMesh;
	readPly(caseRoot + CaseName + "_uv.ply", UVMesh.verts, UVMesh.faceInds);

	//-----uv map to geo  (1 to 1)
	std::vector<int> u_to_g(UVMesh.numV(), -1);
	for (int f = 0; f < UVMesh.numF(); f++)
	{
		cv::Vec3i uface = UVMesh.faceInds[f];
		cv::Vec3i gface = GeoMesh.faceInds[f];
		for (int i = 0; i < 3; i++)
			if (u_to_g[uface[i]] < 0)
				u_to_g[uface[i]] = gface[i];
	} // end for f

	ofstream u_to_g_Stream(caseRoot + CaseName + "_u_to_g.txt");
	int numc = 1;
	for (int vi = 0; vi < UVMesh.numV(); vi++)
		u_to_g_Stream << numc << " " << u_to_g[vi] << endl;
	u_to_g_Stream.close();
	u_to_g_Stream.clear();

	//-----geo map to uv (1 to N)
	std::vector<std::vector<int>> g_to_u(GeoMesh.numV(), std::vector<int>());
	for (int vi = 0; vi < UVMesh.numV(); vi++)
	{
		int gid = u_to_g[vi];
		if (gid < 0)
		{
			printf("%d \n", vi);
		}
		else
			g_to_u[gid].push_back(vi);
	} // end for vi

	ofstream g_to_u_Stream(caseRoot + CaseName + "_g_to_u.txt");
	for (int vi = 0; vi < GeoMesh.numV(); vi++)
	{
		numc = g_to_u[vi].size();
		g_to_u_Stream << numc;
		if (numc < 1)
			printf("invalid g_to_u mapping\n");
		for (int c = 0; c < numc; c++)
			g_to_u_Stream << " " << g_to_u[vi][c];
		g_to_u_Stream << endl;
	}
	g_to_u_Stream.close();
	g_to_u_Stream.clear();

}

void rasterizePixel_withFaceInfo(R_Mesh& UVMesh, int imgH, int imgW, string saveInfo)
{
	UVMesh.scaleMesh(imgH, imgW, 1.);
	RayIntersection myTracer;
	myTracer.addObj(&UVMesh);

	std::vector<cv::Vec2i> Pixel_Valid;
	std::vector<int> InFaceID;
	std::vector<cv::Vec2f> InFaceAB;

	cv::Mat validMask = cv::Mat::zeros(imgH, imgW, CV_32FC3);

	for (int y = 0; y < imgH; y++)
	{
		for (int x = 0; x < imgW; x++)
		{
			cv::Vec3f ori(x, imgH-y, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				InFaceID.push_back(fID);
				InFaceAB.push_back(cv::Vec2f(h.u, h.v));
				Pixel_Valid.push_back(cv::Vec2i(x, y));
				validMask.at<cv::Vec3f>(y, x) = cv::Vec3f(255., 255., 255.);
				
			} // end for else
		} // end for x
	} // end for y

	cv::imwrite(saveInfo + "_uvmask.png", validMask);
	saveSampleInfo_InFaceAB(saveInfo + "_pixelGeoSample.txt", InFaceID, InFaceAB);

	//----save valid pixel
	ofstream pixelStream(saveInfo + "_validPixel.txt");
	pixelStream << "#" << " " << Pixel_Valid.size() << endl;
	for (int i = 0; i < Pixel_Valid.size(); i++)
		pixelStream << Pixel_Valid[i][0] << " " << Pixel_Valid[i][1] << endl;
	
	pixelStream.clear();
	pixelStream.close();
	
}

void GeoImage_Rasterization()
{
	string caseRoot = "./Data/shortskirt/Canonical/weld/";
	string CaseName = "PD30";

	R_Mesh UVMesh;
	readPly(caseRoot + CaseName + "_uv.ply", UVMesh.verts, UVMesh.faceInds);

	int img_Size = 1024;
	char imgSizeCase[8];
	std::snprintf(imgSizeCase, sizeof(imgSizeCase), "%d", img_Size);

	rasterizePixel_withFaceInfo(UVMesh, img_Size, img_Size, caseRoot + "test/" + CaseName + "_" + string(imgSizeCase));
}


cv::Mat Rasterize_FeatInfo(std::vector<cv::Vec2i>& ValidPixel, std::vector<int>& PixelInFace, std::vector<cv::Vec2f>& PixelInAB,
	                       std::vector<cv::Vec3i>& FeatFaceID, std::vector<cv::Vec3f>& Featarray, int imgH, int imgW)
{
	cv::Mat RenderImage = cv::Mat::zeros(imgH, imgW, CV_32FC3);
	int numValidPixel = ValidPixel.size();

#pragma omp parallel for
	for (int i = 0; i < numValidPixel; i++)
	{
		cv::Vec2i XY = ValidPixel[i];
		cv::Vec3i FaceID = FeatFaceID[PixelInFace[i]];
		cv::Vec2f UV = PixelInAB[i];
		cv::Vec3f feature = (1. - UV[0] - UV[1]) * Featarray[FaceID[0]] + UV[0] * Featarray[FaceID[1]] + UV[1] * Featarray[FaceID[2]];
		RenderImage.at<cv::Vec3f>(XY[1], XY[0]) = (cv::Vec3f(feature[2], feature[1], feature[0]) + cv::Vec3f(1., 1., 1.)) * 0.5;
	} // end for i

	return RenderImage;
}

void Rendering_NormalSequencs()
{
	string caseRoot = "./Data/shortskirt/";
	string ResCaseName = "PD10";
	int img_Size = 1024;
	char imgSizeCase[8];
	std::snprintf(imgSizeCase, sizeof(imgSizeCase), "%d", img_Size);

	//-- read ValidPixelXY
	std::vector<cv::Vec2i> ValidPixelXY;
	string XYCaseName = caseRoot + "Canonical/weld/test/" + ResCaseName + "_" + string(imgSizeCase) + "_validPixel.txt";
	read_validPixelFile(XYCaseName, ValidPixelXY);

	
	//--read ValidInFace Information
	std::vector<int> inFaceID;
	std::vector<cv::Vec2f> inFaceUV;
	string inFaceCaseName = caseRoot + "Canonical/weld/test/" + ResCaseName + "_" + string(imgSizeCase) + "_pixelGeoSample.txt";
	read_PixelInFaceFile(inFaceCaseName, inFaceID, inFaceUV);

	assert(ValidPixelXY.size() == inFaceID.size());
	
	//-- read Geo Face
	//std::vector<cv::Vec3f> tempVerts;
	//std::vector<cv::Vec3i> GeoMeshFaces;
	R_Mesh GeoMesh;
	string GeoFaceCase = caseRoot + "Canonical/weld/" + ResCaseName + "_geo.ply";
	readPly(GeoFaceCase, GeoMesh.verts, GeoMesh.faceInds);

	//-- render normal
	string motionCase = "/";
	string vertCase = "/PD10/PD10_";
	string framePref = caseRoot + motionCase + vertCase;

	int frame0 = 10;
	int frame1 = 20;
	for (int frameI = frame0; frameI < frame1 + 1; frameI++)
	{
		char frameName[8];
		std::snprintf(frameName, sizeof(frameName), "%07d", frameI);

		//string frameGeoName = caseRoot + motionCase + ResCaseName + "/" + string(frameName) + ".ply";
		string frameGeoName = framePref + string(frameName) + ".obj";
		//printf("%s", frameGeoName.c_str());

		std::vector<cv::Vec3f> frameGeoVerts;
		std::vector<cv::Vec3f> frameGeoNorms;
		std::vector<cv::Vec3i> frameGeoFace;
		readObjVNArray(frameGeoName, frameGeoVerts, frameGeoNorms);
		//readPly(frameGeoName, frameGeoVerts, frameGeoFace);

		if (frameGeoNorms.size() < 1)
			frameGeoNorms = GeoMesh.calcVertNorm(frameGeoVerts);

		//printf("%d", frameGeoNorms.size());

		cv::Mat normMap = Rasterize_FeatInfo(ValidPixelXY, inFaceID, inFaceUV, GeoMesh.faceInds, frameGeoNorms, img_Size, img_Size);

		char nframeName[8];
		std::snprintf(nframeName, sizeof(nframeName), "%07d", frameI+1);

		string saveName = framePref + "normal_DDE/" + string(nframeName) + ".png";
		//printf("%s", saveName.c_str());
		cv::imwrite(saveName, normMap * 255.);

	} // end for frameI
}

//------------------------------------------------------
// for DDE comparison
//------------------------------------------------------

R_Mesh loadTextMeshes(std::string fName)
{
	R_Mesh model;
	readPly(fName, model.verts, model.faceInds);
	model.bbmin = cv::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	model.bbmax = cv::Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (int i = 0; i < model.numV(); i++)
	{
		cv::Vec3f v = model.verts[i];
		for (int d = 0; d < 3; d++)
		{
			model.bbmin[d] = MIN(model.bbmin[d], v[d]);
			model.bbmax[d] = MAX(model.bbmax[d], v[d]);
		}
	}
	printf("numV: %d, numF: %d\n", model.numV(), model.numF());
	return model;
}

# define TXT_ALPHA 296.437866

void calcTexSize()
{
	std::string F_prefix = "./Data/shortskirt/";
	std::string geoName = F_prefix + "Canonical/weld/PD10_geo.ply";
	R_Mesh geo_model = loadTextMeshes(geoName);
	float geo_avEdgeL = geo_model.calAverageEdgeLen();
	printf("geoLeng: %f\n", geo_avEdgeL);
	std::string uvName = F_prefix + "Canonical/weld/PD10_uv.ply";
	R_Mesh txt_model = loadTextMeshes(uvName);
	float txt_avEdgeL = txt_model.calAverageEdgeLen();
	printf("geo avarage len: %f, txt avarage len: %f\n", geo_avEdgeL, txt_avEdgeL);
	int ImgS = int(TXT_ALPHA * geo_avEdgeL / txt_avEdgeL + 0.5);
	printf("Txt_Size: %d\n", ImgS);
	/*std::string saveName = F_prefix + caseName + "uv/txt_0.txt";
	ofstream txtStream(saveName);
	txtStream << ImgS << endl;
	txtStream.close();
	txtStream.clear();*/
}

cv::Mat blurGapTexture(cv::Mat oriImg, cv::Mat mask, int numIter = 3)
{
	int iter = numIter; // enlarge iter pixels
	cv::Mat mm = mask.clone();
	cv::Mat rstImg = oriImg.clone();
	while ((iter--) > 0)
	{
		cv::Mat iterImg = rstImg.clone();
		cv::Mat iterMask = mm.clone();
		for (int y = 0; y < oriImg.rows; y++)
		{
			for (int x = 0; x < oriImg.cols; x++)
			{
				cv::Vec3f color(0., 0., 0.);
				int cc = 0;
				if (mask.at<int>(y, x) > 0)
					color = oriImg.at<cv::Vec3f>(y, x);
				else
				{
					for (int hy = -1; hy <= 1; hy++)
					{
						for (int hx = -1; hx <= 1; hx++)
						{
							int px = MAX(MIN(x + hx, oriImg.cols - 1), 0);
							int py = MAX(MIN(y + hy, oriImg.rows - 1), 0);
							if (iterMask.at<int>(py, px) > 0)
							{
								color += iterImg.at<cv::Vec3f>(py, px);
								cc += 1;
							}
						} // end for hx
					} // end for hy
					if (cc > 0)
					{
						color = color / float(cc);
						mm.at<int>(y, x) = 255;
					}
				}
				rstImg.at<cv::Vec3f>(y, x) = color;
			} // end for x
		} //end for y
	}
	return rstImg;
}

void grab_normal_from_nimg()
{
	std::string caseName = "0_House_Dancing/";
	std::string F_prefix = "./Data/shortskirt/";
	std::string uvName = F_prefix + "Canonical/weld/PD10_uv.ply";
	int imgS = 649;
	std::string uvMaskName = F_prefix + "Canonical/weld/test/PD10_649_uvmask.png";
	cv::Mat maskImg = cv::imread(uvMaskName, cv::IMREAD_GRAYSCALE);
	maskImg.convertTo(maskImg, CV_32SC1);

	R_Mesh UVMesh;
	readPly(uvName, UVMesh.verts, UVMesh.faceInds);

	std::string nimg_root = "./Data/shortskirt/0_House_Dancing/";
	std::string save_root = nimg_root + "normal_obj/";
	
	int frame0 = 21;
	int frame1 = 321;
	for (int frameI = frame0; frameI < frame1 + 1; frameI++)
	{
		char frameName[8];
		std::snprintf(frameName, sizeof(frameName), "%07d", frameI);

		std::string imgName = nimg_root + "normal_infer/" + string(frameName) + ".png";
		cv::Mat nimg = cv::imread(imgName, cv::IMREAD_COLOR);
		nimg.convertTo(nimg, CV_32FC3);
		nimg = blurGapTexture(nimg, maskImg);
		//cv::imwrite("./test/blue.png", nimg);

		std::vector<cv::Vec3f> normCArray(UVMesh.numV(), cv::Vec3f(0., 0., 0.));
		for (int vi = 0; vi < UVMesh.numV(); vi++)
		{
			cv::Vec3f vpos = UVMesh.verts[vi];
			cv::Vec2f imgPos = cv::Vec2f(vpos[0] * imgS, (1. - vpos[1]) * imgS);
			cv::Vec3f normV = getInfoFromMat_3f(imgPos, nimg) / 255.;
			normV = cv::Vec3f(normV[2], normV[1], normV[0]) * 2. - cv::Vec3f(1., 1., 1.);
			normCArray[vi] = normalize(normV);
		}// end for vi
		std::string saveName = save_root + string(frameName) + ".obj";
		saveObjFile(saveName, UVMesh.verts, normCArray);
	}// end for frameI
}

void LabelMateraial()
{
	string caseRoot = "./Data/shortskirt/Canonical/weld/";
	string CaseName = "PD30";

	/*R_Mesh GeoMesh;
	readPly(caseRoot + CaseName + "_geo.ply", GeoMesh.verts, GeoMesh.faceInds);*/
	R_Mesh UVMesh;
	readPly(caseRoot + CaseName + "_uv.ply", UVMesh.verts, UVMesh.faceInds);

	cv::Mat label_mask = cv::imread(caseRoot + "test/color_1024_uvmask.jpg", cv::IMREAD_COLOR);
	label_mask.convertTo(label_mask, CV_32FC3);
	cv::Mat valid_mask = cv::imread(caseRoot + "test/PD10_1024_uvmask.png", cv::IMREAD_GRAYSCALE);
	valid_mask.convertTo(valid_mask, CV_32SC1);
	label_mask = blurGapTexture(label_mask, valid_mask);

	int imgS = 1024;
	std::vector<cv::Vec3i> label_colors(UVMesh.numV(), cv::Vec3i(0, 0, 0));
	for (int vi = 0; vi < UVMesh.numV(); vi++)
	{
		cv::Vec3f vpos = UVMesh.verts[vi];
		cv::Vec2f imgPos = cv::Vec2f(vpos[0] * imgS, (1. - vpos[1]) * imgS);
		cv::Vec3f tagColor = getInfoFromMat_3f(imgPos, label_mask) / 255.;
		if (tagColor[0] > 0.35)
			label_colors[vi][2] = 255;
		else
		{
			if (tagColor[2] > 0.35)
				label_colors[vi][0] = 255;
			else
				label_colors[vi][1] = 255;
		}
	}
	savePlyFile(caseRoot + CaseName + "_label_uv.ply", UVMesh.verts, label_colors, UVMesh.faceInds);
}

int main()
{
	//--- step_1: set up sampling between different resolutions across UV
	//-- To do: Bi-dir sampling
	Sampling_between_Different_PDResolution_Across_UV();
	//Geo_UV_Map();


	//-- step_2: gen. uv rendering rasteration
	//GeoImage_Rasterization();

	//LabelMateraial();

	//Rendering_NormalSequencs();

	//BodySeed_Sampling_Based_ColorMask();

	//------------------------------------------------------
	// for DDE comparison
	//------------------------------------------------------
	//calcTexSize();
	//grab_normal_from_nimg();

	printf("Done.");
	while (1);
}