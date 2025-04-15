// Globe3D.js
import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import Globe from 'three-globe';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const Globe3D = () => {
  const globeContainerRef = useRef();
  const globeRef = useRef();
  const [selectedIndustry, setSelectedIndustry] = useState("Technology, Advertising, Media, and Information");
  const [topCities, setTopCities] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState([]);

  const updateGlobePoints = useCallback((formatted) => {
    if (!globeRef.current || !formatted || formatted.length === 0) return;
    
    console.log("🌍 Updating globe points:", {
      totalPoints: formatted.length,
      pointSample: formatted.slice(0, 3),
      citiesIncluded: formatted.map(d => d.city).join(", ")
    });
    
    // Wait a frame before setting new points
    requestAnimationFrame(() => {
      try {
        globeRef.current
          .pointsData(formatted)
          .pointAltitude(0.03)  // Increased altitude further
          .pointRadius('radius')
          .pointResolution(64)
          .pointColor(() => 'rgb(0, 87, 255)')  // Removed transparency
          .pointsMerge(false)
          .pointsTransitionDuration(1000);

        console.log("✅ Points updated on globe");
      } catch (error) {
        console.error("Error updating globe points:", error);
      }
    });
  }, []);

  const fetchClustersForIndustry = useCallback((industry) => {
    setLoading(true);
    setError(null);
    console.log("🔍 Fetching data for industry:", industry);

    fetch(`http://localhost:5001/api/city-density?industry=${encodeURIComponent(industry)}`)
      .then(res => {
        if (!res.ok) {
          throw new Error(`Server responded with status: ${res.status}`);
        }
        return res.json();
      })
      .then(raw => {
        console.log("📊 Raw data details:", {
          dataLength: raw.length,
          sampleData: raw.slice(0, 3),
          uniqueCities: [...new Set(raw.map(d => d.market))].length
        });

        if (!Array.isArray(raw) || raw.length === 0) {
          throw new Error("No data returned from server");
        }

        const formatted = raw.map(d => ({
          lat: parseFloat(d.lat),
          lng: parseFloat(d.lon),
          city: d.market,
          count: d.count,
          radius: Math.max(0.25, Math.log(d.count + 1) * 0.4)  // Slightly smaller radius
        }));

        // Log the top 5 cities by company count
        const topCitiesByCount = [...formatted]
          .sort((a, b) => b.count - a.count)
          .slice(0, 5);
        
        console.log("🏢 Top 5 cities for", industry, ":", 
          topCitiesByCount.map(city => `${city.city}: ${city.count} companies`));

        setTopCities(topCitiesByCount);
        setData(formatted);
        updateGlobePoints(formatted);
      })
      .catch(err => {
        console.error("❌ Error fetching data:", err);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, [updateGlobePoints]);

  // Fetch data when industry changes
  useEffect(() => {
    fetchClustersForIndustry(selectedIndustry);
  }, [selectedIndustry, fetchClustersForIndustry]);

  // Initialize the globe
  useEffect(() => {
    if (!globeContainerRef.current) return;

    console.log("Setting up 3D scene");
    
    // Scene setup
    const scene = new THREE.Scene();
    
    // Camera setup
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(200, 150, 200);
    camera.lookAt(0, 0, 0);

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true,
      canvas: globeContainerRef.current
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 150;
    controls.maxDistance = 500;
    controls.rotateSpeed = 0.5;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0.5;

    // Globe setup
    const globe = new Globe()
      .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
      .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
      .pointsData([])
      .pointRadius('radius')
      .pointResolution(64)
      .pointColor(() => 'rgb(0, 87, 255)')  // Removed transparency
      .pointAltitude(0.06)  // Increased altitude further
      .pointsMerge(false)
      .pointsTransitionDuration(1000);

    globeRef.current = globe;
    scene.add(globe);

    // Load and add US states borders
    fetch('https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_1_states_provinces_shp.geojson')
      .then(res => res.json())
      .then(states => {
        const usStates = states.features.filter(d => d.properties.admin === 'United States of America');
        globe
          .polygonsData(usStates)
          .polygonAltitude(0.001)  // Very minimal altitude for state lines
          .polygonCapColor(() => 'rgba(200, 200, 200, 0)')  // Transparent fill
          .polygonSideColor(() => 'rgba(150, 150, 150, 0)')  // Transparent sides
          .polygonStrokeColor(() => '#ffffff')  // White state lines
          .polygonStrokeWidth(0.8);  // Slightly thicker lines for visibility
      })
      .catch(error => console.error('Error loading US states data:', error));

    // Add ambient and directional light for better bubble effect
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Add point light for specular highlights
    const pointLight = new THREE.PointLight(0xffffff, 0.5);
    pointLight.position.set(-100, 200, 100);
    scene.add(pointLight);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      const { innerWidth, innerHeight } = window;
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    };
    window.addEventListener('resize', handleResize);

    // If we have data already, update the points
    if (data.length > 0) {
      updateGlobePoints(data);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      scene.remove(globe);
      controls.dispose();
    };
  }, []); // Empty dependency array as we only want to initialize once

  return (
    <>
      <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 1000 }}>
        <div style={{ 
          backgroundColor: 'rgba(8, 8, 24, 0.85)',
          padding: '20px',
          borderRadius: '12px',
          color: '#fff',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px 0 rgba(0, 87, 255, 0.1)',
        }}>
          <h3 style={{ 
            margin: '0 0 15px 0',
            fontSize: '18px',
            fontWeight: '500',
            color: '#00f7ff',
            textTransform: 'uppercase',
            letterSpacing: '2px'
          }}>Select Industry</h3>
          <select 
            value={selectedIndustry} 
            onChange={(e) => setSelectedIndustry(e.target.value)}
            style={{
              padding: '12px 16px',
              borderRadius: '8px',
              border: '1px solid rgba(0, 247, 255, 0.3)',
              backgroundColor: 'rgba(0, 0, 0, 0.3)',
              color: '#fff',
              fontSize: '14px',
              width: '300px',
              outline: 'none',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 0 15px rgba(0, 247, 255, 0.1)'
            }}
          >
            <option value="Technology, Advertising, Media, and Information">Technology, Advertising, Media, and Information</option>
            <option value="Legal Services">Legal Services</option>
            <option value="Financial Services and Insurance">Financial Services and Insurance</option>
          </select>
          <p style={{ 
            fontSize: '13px', 
            marginTop: '12px',
            color: 'rgba(0, 247, 255, 0.7)',
            letterSpacing: '1px'
          }}>
            {data.length} cities found
          </p>
        </div>
      </div>

      {topCities.length > 0 && (
        <div style={{ 
          position: 'absolute', 
          top: 20, 
          right: 20, 
          zIndex: 1000,
          backgroundColor: 'rgba(8, 8, 24, 0.85)',
          padding: '20px',
          borderRadius: '12px',
          color: '#fff',
          maxWidth: '300px',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px 0 rgba(0, 87, 255, 0.1)'
        }}>
          <h3 style={{ 
            margin: '0 0 15px 0',
            fontSize: '18px',
            fontWeight: '500',
            color: '#00f7ff',
            textTransform: 'uppercase',
            letterSpacing: '2px'
          }}>Top 5 Cities Leased</h3>
          <ul style={{ padding: 0, margin: 0, listStyle: 'none' }}>
            {topCities.map((city, index) => (
              <li key={index} style={{ 
                marginBottom: '10px',
                padding: '12px',
                backgroundColor: 'rgba(0, 247, 255, 0.05)',
                borderRadius: '8px',
                display: 'flex',
                justifyContent: 'space-between',
                border: '1px solid rgba(0, 247, 255, 0.1)',
                transition: 'all 0.3s ease',
                cursor: 'default',
                ':hover': {
                  backgroundColor: 'rgba(0, 247, 255, 0.1)',
                  transform: 'translateY(-2px)'
                }
              }}>
                <span style={{ color: 'rgba(255, 255, 255, 0.9)' }}>{city.city}</span>
                <span style={{ 
                  fontWeight: '500',
                  color: '#00f7ff'
                }}>{city.count} companies</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div style={{ 
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(255, 0, 0, 0.15)',
          color: '#ff4444',
          padding: '20px 40px',
          borderRadius: '12px',
          zIndex: 1000,
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255, 0, 0, 0.3)',
          boxShadow: '0 0 20px rgba(255, 0, 0, 0.2)'
        }}>
          Error: {error}
        </div>
      )}

      {loading && (
        <div style={{ 
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(8, 8, 24, 0.85)',
          color: '#00f7ff',
          padding: '20px 40px',
          borderRadius: '12px',
          zIndex: 1000,
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(0, 247, 255, 0.3)',
          boxShadow: '0 0 20px rgba(0, 247, 255, 0.2)'
        }}>
          <span style={{ 
            letterSpacing: '2px',
            textTransform: 'uppercase',
            fontSize: '14px'
          }}>Loading...</span>
        </div>
      )}

      {/* Add title in lower right corner */}
      <div style={{ 
        position: 'absolute', 
        bottom: 30, 
        right: 30, 
        zIndex: 1000,
        backgroundColor: 'rgba(8, 8, 24, 0.85)',
        padding: '20px 30px',
        borderRadius: '12px',
        color: '#fff',
        backdropFilter: 'blur(8px)',
        border: '1px solid rgba(0, 247, 255, 0.3)',
        boxShadow: '0 8px 32px 0 rgba(0, 87, 255, 0.1)',
        transform: 'translateZ(0)'
      }}>
        <h2 style={{ 
          margin: 0,
          fontSize: '32px',  // Increased from 24px
          fontWeight: '600',
          color: '#00f7ff',
          textTransform: 'uppercase',
          letterSpacing: '3px',
          textShadow: '0 0 15px rgba(0, 247, 255, 0.5)',  // Increased glow
          display: 'flex',
          alignItems: 'center',
          gap: '12px'  // Increased gap
        }}>
          <span style={{ fontSize: '74px', marginRight: '8px' }}>⬡</span>
          3D Lease Density Map
        </h2>
      </div>

      <canvas ref={globeContainerRef} style={{ width: '100vw', height: '100vh' }} />
    </>
  );
};

export default Globe3D;

      

