__device__ Vec3f renderKernel(curandState* randstate, const float4* HDRmap, const float4* gpuNodes, const float4* gpuTriWoops,
    const float4* gpuDebugTris, const int* gpuTriIndices, Vec3f& rayorig, Vec3f& raydir, unsigned int leafcount, unsigned int tricount)
{
    Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f); // colour mask
    Vec3f accucolor = Vec3f(0.0f, 0.0f, 0.0f); // accumulated colour
    Vec3f direct = Vec3f(0, 0, 0);

    VolumetricProps volProps;

    for (int bounces = 0; bounces < 100; bounces++){  // iteration up to 4 bounces (instead of recursion in CPU code)

        int hitSphereIdx = -1;
        int hitTriIdx = -1;
        int bestTriIdx = -1;
        int geomtype = -1;
        float hitSphereDist = 1e10;
        float hitDistance = 1e10;
        float scene_t = 1e10;
        Vec3f objcol = Vec3f(0, 0, 0);
        Vec3f emit = Vec3f(0, 0, 0);
        Vec3f hitpoint; // intersection point
        Vec3f n; // normal
        Vec3f nl; // oriented normal
        Vec3f nextdir; // ray direction of next path segment
        Vec3f trinormal = Vec3f(0, 0, 0);
        Vec3f shadenormal = Vec3f(0, 0, 0);
        Refl_t refltype;
        float ray_tmin = 0.01f;// 0.00001f; // set to 0.01f when using refractive material
        float ray_tmax = 1e10;

        // intersect all triangles in the scene stored in BVH

        int debugbingo = 0;

        intersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin), make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
            gpuNodes, gpuTriWoops, gpuDebugTris, gpuTriIndices, bestTriIdx, hitDistance, debugbingo, trinormal, shadenormal, leafcount, tricount, false);

        //DEBUGintersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin), make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
        //gpuNodes, gpuTriWoops, gpuDebugTris, gpuTriIndices, bestTriIdx, hitDistance, debugbingo, trinormal, leafcount, tricount, false);


#if 0
        // intersect all spheres in the scene
        float numspheres = sizeof(spheres) / sizeof(Sphere);
        for (int i = int(numspheres); i--;)  // for all spheres in scene
            // keep track of distance from origin to closest intersection point
            if ((hitSphereDist = spheres[i].intersect(makeRay(rayorig, raydir))) && hitSphereDist < scene_t && hitSphereDist > 0.01f){
                scene_t = hitSphereDist; hitSphereIdx = i; geomtype = 1;
            }
#endif

        if (hitDistance < scene_t && hitDistance > ray_tmin) // triangle hit
        {
            scene_t = hitDistance;
            hitTriIdx = bestTriIdx;
            geomtype = 2;
        }

        // sky gradient colour
        //float t = 0.5f * (raydir.y + 1.2f);
        //Vec3f skycolor = Vec3f(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3f(0.9f, 0.3f, 0.0f) * t;

#ifdef HDR
        // HDR 

        if (scene_t > 1e9) { // if ray misses scene, return sky

            // HDR environment map code based on Syntopia "Path tracing 3D fractals"
            // http://blog.hvidtfeldts.net/index.php/2015/01/path-tracing-3d-fractals/
            // https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/IBL-Pathtracer.frag
            // GLSL code: 
            // vec3 equirectangularMap(sampler2D sampler, vec3 dir) {
            //		dir = normalize(dir);
            //		vec2 longlat = vec2(atan(dir.y, dir.x) + RotateMap, acos(dir.z));
            //		return texture2D(sampler, longlat / vec2(2.0*PI, PI)).xyz; }

            // Convert (normalized) dir to spherical coordinates.
            // float longlatX = atan2f(raydir.x, raydir.z); // Y is up, swap x for y and z for x
            // longlatX = longlatX < 0.f ? longlatX + TWO_PI : longlatX;  // wrap around full circle if negative
            // float longlatX = atan2f(raydir.z, raydir.x); // Y is up, swap x for y and z for x
            // longlatX = raydir.z < 0.f ? longlatX + TWO_PI : longlatX;  // wrap around full circle if negative
            float longlatX = ((raydir.x > 0 ? atanf(raydir.z / raydir.x) : atanf(raydir.z / raydir.x) + M_PI) + M_PI * 0.5);
            float longlatY = acosf(raydir.y); // add RotateMap at some point, see Fragmentarium

            // map theta and phi to u and v texturecoordinates in [0,1] x [0,1] range
            float offsetY = 0.5f;
            float u = longlatX / TWO_PI; // +offsetY;
            float v = longlatY / M_PI;

            // map u, v to integer coordinates
            int u2 = (int)(u * HDRwidth); //% HDRwidth;
            int v2 = (int)(v * HDRheight); // % HDRheight;

            // compute the texel index in the HDR map 
            int HDRtexelidx = u2 + v2 * HDRwidth;

            //float4 HDRcol = HDRmap[HDRtexelidx];
            float4 HDRcol = tex1Dfetch(HDRtexture, HDRtexelidx);  // fetch from texture
            Vec3f HDRcol2 = Vec3f(HDRcol.x, HDRcol.y, HDRcol.z);

            // 			emit = HDRcol2 * 2.0f;
            emit = HDRcol2;
            accucolor += (mask * emit);
            return accucolor;
        }

#endif // end of HDR

        // SPHERES:
        if (geomtype == 1){
            Sphere &hitsphere = spheres[hitSphereIdx]; // hit object with closest intersection
            hitpoint = rayorig + raydir * scene_t;  // intersection point on object
            n = Vec3f(hitpoint.x - hitsphere.pos.x, hitpoint.y - hitsphere.pos.y, hitpoint.z - hitsphere.pos.z);	// normal
            n.normalize();
            nl = dot(n, raydir) < 0 ? n : n * -1; // correctly oriented normal
            objcol = Vec3f(hitsphere.col.x, hitsphere.col.y, hitsphere.col.z);   // object colour
            emit = Vec3f(hitsphere.emi.x, hitsphere.emi.y, hitsphere.emi.z);  // object emission
            refltype = hitsphere.refl;
            accucolor += (mask * emit);
        }

        // TRIANGLES:
        if (geomtype == 2){

            //pBestTri = &pTriangles[triangle_id];
            hitpoint = rayorig + raydir * scene_t; // intersection point

            // float4 normal = tex1Dfetch(triNormalsTexture, pBestTriIdx);	
            n = trinormal;
            n.normalize();
            nl = dot(n, raydir) < 0 ? n : n * -1;  // correctly oriented normal
            //Vec3f colour = hitTriIdx->_colorf;
            // 			Vec3f colour = Vec3f(0.9f, 0.3f, 0.0f); // hardcoded triangle colour  .9f, 0.3f, 0.0f
            Vec3f colour = Vec3f(1.0f, 1.0f, 1.0f); // hardcoded triangle colour  .9f, 0.3f, 0.0f
            // 			refltype = COAT; // objectmaterial
            // 			refltype = REFR; // objectmaterial
            refltype = VOLUME; // objectmaterial
            objcol = colour;
            emit = Vec3f(0.0, 0.0, 0);  // object emission
            accucolor += (mask * emit);
        }

        // SSS
        if (dot(n, nl) < 0) {

            float e0 = curand_uniform(randstate), e1 = curand_uniform(randstate), e2 = curand_uniform(randstate);
            float distToExit = scene_t;
            float s_free;//free path length

            // is it right?? does it bring correlation??
            //             int channel;
            //             if (e0 < e1 && e0 < e2) {
            //                 channel = 0;
            //             }
            //             else if (e1 < e2) {
            //                 channel = 1;
            //             }
            //             else{
            //                 channel = 2;
            //             }
            float e3 = curand_uniform(randstate);
            int channel = e3 < 0.333333333333333 ? 0 : e3 < 0.666666666666667 ? 1 : 2;

            Vec3f sigmat = volProps.reduced_sigma_t;
            float sigma0 = sigmat._v[channel];

            float scatter_dist = -log(e0) / sigma0;
            bool success = true;
            if (scatter_dist >= scene_t) {
                scatter_dist = scene_t;
                success = false;
            }
            float pdfFailure = 0;
            float pdfSuccess = 0;
            for (int ii = 0; ii < 3; ++ii) {
                float tmp = expf(-sigmat._v[ii] * scatter_dist);
                pdfFailure += tmp;
                pdfSuccess += sigmat._v[ii] * tmp;
            }
            pdfFailure *= (1.0f / 3.0f);
            pdfSuccess *= (1.0f / 3.0f);
            if (success) {
                rayorig = rayorig + raydir * scatter_dist;
                Vec3f dir = volProps.sampleHG(0.7f, e1, e2);  // note that the reduced sigma = sigma * (1 - g)
                Vec3f u, v;
                volProps.generateOrthoBasis(u, v, raydir);
                raydir = (u * dir.x + v * dir.y + raydir * dir.z).normalize();
                mask *= (sigmat * volProps.ss_albedo) * expf(sigmat * -scatter_dist) / pdfSuccess;
                continue;
            }
            else{
                mask *= expf(sigmat * -scatter_dist) / pdfFailure;
            }

            //             Ray sRay = volProps.scatter(makeRay(rayorig, raydir), sigma, s_free, e0, e1, e2);
            //             if (s_free < distToExit) {
            //                 rayorig = Vec3f(sRay.orig.x, sRay.orig.y, sRay.orig.z);
            //                 raydir = Vec3f(sRay.dir.x, sRay.dir.y, sRay.dir.z);
            //                 mask *= (volProps.ss_albedo) * (sigmat / sigma) * expf((sigmat-Vec3f(sigma,sigma,sigma)) * -scene_t)
            //                    ;
            // //                 mask *= volProps.ss_albedo
            // //                     * (sigmat / sigmat)
            // //                     * expf((sigmat - sigmat) * -scene_t)
            // //                     ;
            // //                     / (sigma * expf(sigma * -scene_t));
            //                 continue;//random walk
            //             }
        }

        // basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)

        // diffuse material, based on smallpt by Kevin Beason 
        if (refltype == DIFF){

            // pick two random numbers
            float phi = 2 * M_PI * curand_uniform(randstate);
            float r2 = curand_uniform(randstate);
            float r2s = sqrtf(r2);

            // compute orthonormal coordinate frame uvw with hitpoint as origin 
            Vec3f w = nl; w.normalize();
            Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
            Vec3f v = cross(w, u);

            // compute cosine weighted random ray direction on hemisphere 
            nextdir = u*cosf(phi)*r2s + v*sinf(phi)*r2s + w*sqrtf(1 - r2);
            nextdir.normalize();

            // offset origin next path segment to prevent self intersection
            hitpoint += nl * 0.001f; // scene size dependent

            // multiply mask with colour of object
            mask *= objcol;

        } // end diffuse material

        // Phong metal material from "Realistic Ray Tracing", P. Shirley
        if (refltype == METAL){

            // compute random perturbation of ideal reflection vector
            // the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
            float phi = 2 * M_PI * curand_uniform(randstate);
            float r2 = curand_uniform(randstate);
            float phongexponent = 30;
            float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
            float sinTheta = sqrtf(1 - cosTheta * cosTheta);

            // create orthonormal basis uvw around reflection vector with hitpoint as origin 
            // w is ray direction for ideal reflection
            Vec3f w = raydir - n * 2.0f * dot(n, raydir); w.normalize();
            Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
            Vec3f v = cross(w, u); // v is already normalised because w and u are normalised

            // compute cosine weighted random ray direction on hemisphere 
            nextdir = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
            nextdir.normalize();

            // offset origin next path segment to prevent self intersection
            hitpoint += nl * 0.0001f;  // scene size dependent

            // multiply mask with colour of object
            mask *= objcol;
        }

        // ideal specular reflection (mirror) 
        if (refltype == SPEC){

            // compute relfected ray direction according to Snell's law
            nextdir = raydir - n * dot(n, raydir) * 2.0f;
            nextdir.normalize();

            // offset origin next path segment to prevent self intersection
            hitpoint += nl * 0.001f;

            // multiply mask with colour of object
            mask *= objcol;
        }


        // COAT material based on https://github.com/peterkutz/GPUPathTracer
        // randomly select diffuse or specular reflection
        // looks okay-ish but inaccurate (no Fresnel calculation yet)
        if (refltype == COAT){

            float rouletteRandomFloat = curand_uniform(randstate);
            float threshold = 0.05f;
            Vec3f specularColor = Vec3f(1, 1, 1);  // hard-coded
            bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_Vec3f(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

            if (reflectFromSurface) { // calculate perfectly specular reflection

                // Ray reflected from the surface. Trace a ray in the reflection direction.
                // TODO: Use Russian roulette instead of simple multipliers! 
                // (Selecting between diffuse sample and no sample (absorption) in this case.)

                mask *= specularColor;
                nextdir = raydir - n * 2.0f * dot(n, raydir);
                nextdir.normalize();

                // offset origin next path segment to prevent self intersection
                hitpoint += nl * 0.001f; // scene size dependent
            }

            else {  // calculate perfectly diffuse reflection

                float r1 = 2 * M_PI * curand_uniform(randstate);
                float r2 = curand_uniform(randstate);
                float r2s = sqrtf(r2);

                // compute orthonormal coordinate frame uvw with hitpoint as origin 
                Vec3f w = nl; w.normalize();
                Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
                Vec3f v = cross(w, u);

                // compute cosine weighted random ray direction on hemisphere 
                nextdir = u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2);
                nextdir.normalize();

                // offset origin next path segment to prevent self intersection
                hitpoint += nl * 0.001f;  // // scene size dependent

                // multiply mask with colour of object
                mask *= objcol;
            }
        } // end COAT

        // perfectly refractive material (glass, water)
        // set ray_tmin to 0.01 when using refractive material
        if (refltype == REFR){

            bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
            float nc = 1.0f;  // Index of Refraction air
            float nt = 1.49f;  // Index of Refraction glass/water
            float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
            float ddn = dot(raydir, nl);
            float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

            if (cos2t < 0.0f) // total internal reflection 
            {
                nextdir = raydir - n * 2.0f * dot(n, raydir);
                nextdir.normalize();

                // offset origin next path segment to prevent self intersection
                hitpoint += nl * 0.001f; // scene size dependent
            }
            else // cos2t > 0
            {
                // compute direction of transmission ray
                Vec3f tdir = raydir * nnt;
                //tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
                tdir -= nl * ((ddn*nnt + sqrtf(cos2t)));
                tdir.normalize();

//                 float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc); // BUG
                float R0 = ((nt - nc)*(nt - nc)) / ((nt + nc)*(nt + nc)); // fixed
                float c = 1.f - (into ? -ddn : dot(tdir, n));
                float Re = R0 + (1.f - R0) * c * c * c * c * c;
                float Tr = 1 - Re; // Transmission
                float P = .25f + .5f * Re;
                float RP = Re / P;
                float TP = Tr / (1.f - P);

                // randomly choose reflection or transmission ray
                if (curand_uniform(randstate) < P) // reflection ray
                {
                    mask *= RP;
                    nextdir = raydir - n * 2.0f * dot(n, raydir);
                    nextdir.normalize();

                    hitpoint += nl * 0.001f; // scene size dependent
                }
                else // transmission ray
                {
                    mask *= TP;
                    nextdir = tdir;
                    nextdir.normalize();

                    hitpoint += nl * 0.001f; // epsilon must be small to avoid artefacts
                }
            }
        }

        if (refltype == VOLUME){

            bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?

            // beer lambert
            //             if (!into) {
            //                 mask *= expf(Vec3f(-10 * scene_t, -2 * scene_t, -3 * scene_t));
            //             }

            float nc = 1.0f;  // Index of Refraction air
            float nt = 1.49f;  // Index of Refraction glass/water
            float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
            float ddn = dot(raydir, nl);
            float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

            if (cos2t < 0.0f) // total internal reflection 
            {
                nextdir = raydir - n * 2.0f * dot(n, raydir);
                nextdir.normalize();

                // offset origin next path segment to prevent self intersection
                hitpoint += nl * 0.001f; // scene size dependent
            }
            else // cos2t > 0
            {
                // compute direction of transmission ray
                Vec3f tdir = raydir * nnt;
                //tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
                tdir -= nl * ((ddn*nnt + sqrtf(cos2t)));
                tdir.normalize();

//                 float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc); // BUG
                float R0 = ((nt - nc)*(nt - nc)) / ((nt + nc)*(nt + nc)); // fixed
                float c = 1.f - (into ? -ddn : dot(tdir, n));
                float Re = R0 + (1.f - R0) * c * c * c * c * c;
                float Tr = 1 - Re; // Transmission
                float P = .25f + .5f * Re;
                float RP = Re / P;
                float TP = Tr / (1.f - P);

                // randomly choose reflection or transmission ray
                if (curand_uniform(randstate) < P) // reflection ray
                {
                    mask *= RP;
                    nextdir = raydir - n * 2.0f * dot(n, raydir);
                    nextdir.normalize();

                    hitpoint += nl * 0.001f; // scene size dependent
                }
                else // transmission ray
                {
                    mask *= TP;
                    nextdir = tdir;
                    nextdir.normalize();

                    hitpoint += nl * 0.001f; // epsilon must be small to avoid artefacts
                }
            }
        }

        // set up origin and direction of next path segment
        rayorig = hitpoint;
        raydir = nextdir;
    } // end bounces for loop

    return accucolor;
}

