"use strict";/*!--------------------------------------------------------
 * Copyright (C) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------*/const _amdLoaderGlobal=this,_commonjsGlobal=typeof global=="object"?global:{};var AMDLoader;(function(u){u.global=_amdLoaderGlobal;class _{get isWindows(){return this._detect(),this._isWindows}get isNode(){return this._detect(),this._isNode}get isElectronRenderer(){return this._detect(),this._isElectronRenderer}get isWebWorker(){return this._detect(),this._isWebWorker}get isElectronNodeIntegrationWebWorker(){return this._detect(),this._isElectronNodeIntegrationWebWorker}constructor(){this._detected=!1,this._isWindows=!1,this._isNode=!1,this._isElectronRenderer=!1,this._isWebWorker=!1,this._isElectronNodeIntegrationWebWorker=!1}_detect(){this._detected||(this._detected=!0,this._isWindows=_._isWindows(),this._isNode=typeof module<"u"&&!!module.exports,this._isElectronRenderer=typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.electron<"u"&&process.type==="renderer",this._isWebWorker=typeof u.global.importScripts=="function",this._isElectronNodeIntegrationWebWorker=this._isWebWorker&&typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.electron<"u"&&process.type==="worker")}static _isWindows(){return typeof navigator<"u"&&navigator.userAgent&&navigator.userAgent.indexOf("Windows")>=0?!0:typeof process<"u"?process.platform==="win32":!1}}u.Environment=_})(AMDLoader||(AMDLoader={}));var AMDLoader;(function(u){class _{constructor(r,c,a){this.type=r,this.detail=c,this.timestamp=a}}u.LoaderEvent=_;class m{constructor(r){this._events=[new _(1,"",r)]}record(r,c){this._events.push(new _(r,c,u.Utilities.getHighPerformanceTimestamp()))}getEvents(){return this._events}}u.LoaderEventRecorder=m;class f{record(r,c){}getEvents(){return[]}}f.INSTANCE=new f,u.NullLoaderEventRecorder=f})(AMDLoader||(AMDLoader={}));var AMDLoader;(function(u){class _{static fileUriToFilePath(f,h){if(h=decodeURI(h).replace(/%23/g,"#"),f){if(/^file:\/\/\//.test(h))return h.substr(8);if(/^file:\/\//.test(h))return h.substr(5)}else if(/^file:\/\//.test(h))return h.substr(7);return h}static startsWith(f,h){return f.length>=h.length&&f.substr(0,h.length)===h}static endsWith(f,h){return f.length>=h.length&&f.substr(f.length-h.length)===h}static containsQueryString(f){return/^[^\#]*\?/gi.test(f)}static isAbsolutePath(f){return/^((http:\/\/)|(https:\/\/)|(file:\/\/)|(\/))/.test(f)}static forEachProperty(f,h){if(f){let r;for(r in f)f.hasOwnProperty(r)&&h(r,f[r])}}static isEmpty(f){let h=!0;return _.forEachProperty(f,()=>{h=!1}),h}static recursiveClone(f){if(!f||typeof f!="object"||f instanceof RegExp||!Array.isArray(f)&&Object.getPrototypeOf(f)!==Object.prototype)return f;let h=Array.isArray(f)?[]:{};return _.forEachProperty(f,(r,c)=>{c&&typeof c=="object"?h[r]=_.recursiveClone(c):h[r]=c}),h}static generateAnonymousModule(){return"===anonymous"+_.NEXT_ANONYMOUS_ID+++"==="}static isAnonymousModule(f){return _.startsWith(f,"===anonymous")}static getHighPerformanceTimestamp(){return this.PERFORMANCE_NOW_PROBED||(this.PERFORMANCE_NOW_PROBED=!0,this.HAS_PERFORMANCE_NOW=u.global.performance&&typeof u.global.performance.now=="function"),this.HAS_PERFORMANCE_NOW?u.global.performance.now():Date.now()}}_.NEXT_ANONYMOUS_ID=1,_.PERFORMANCE_NOW_PROBED=!1,_.HAS_PERFORMANCE_NOW=!1,u.Utilities=_})(AMDLoader||(AMDLoader={}));var AMDLoader;(function(u){function _(h){if(h instanceof Error)return h;const r=new Error(h.message||String(h)||"Unknown Error");return h.stack&&(r.stack=h.stack),r}u.ensureError=_;class m{static validateConfigurationOptions(r){function c(a){if(a.phase==="loading"){console.error('Loading "'+a.moduleId+'" failed'),console.error(a),console.error("Here are the modules that depend on it:"),console.error(a.neededBy);return}if(a.phase==="factory"){console.error('The factory function of "'+a.moduleId+'" has thrown an exception'),console.error(a),console.error("Here are the modules that depend on it:"),console.error(a.neededBy);return}}if(r=r||{},typeof r.baseUrl!="string"&&(r.baseUrl=""),typeof r.isBuild!="boolean"&&(r.isBuild=!1),typeof r.paths!="object"&&(r.paths={}),typeof r.config!="object"&&(r.config={}),typeof r.catchError>"u"&&(r.catchError=!1),typeof r.recordStats>"u"&&(r.recordStats=!1),typeof r.urlArgs!="string"&&(r.urlArgs=""),typeof r.onError!="function"&&(r.onError=c),Array.isArray(r.ignoreDuplicateModules)||(r.ignoreDuplicateModules=[]),r.baseUrl.length>0&&(u.Utilities.endsWith(r.baseUrl,"/")||(r.baseUrl+="/")),typeof r.cspNonce!="string"&&(r.cspNonce=""),typeof r.preferScriptTags>"u"&&(r.preferScriptTags=!1),r.nodeCachedData&&typeof r.nodeCachedData=="object"&&(typeof r.nodeCachedData.seed!="string"&&(r.nodeCachedData.seed="seed"),(typeof r.nodeCachedData.writeDelay!="number"||r.nodeCachedData.writeDelay<0)&&(r.nodeCachedData.writeDelay=1e3*7),!r.nodeCachedData.path||typeof r.nodeCachedData.path!="string")){const a=_(new Error("INVALID cached data configuration, 'path' MUST be set"));a.phase="configuration",r.onError(a),r.nodeCachedData=void 0}return r}static mergeConfigurationOptions(r=null,c=null){let a=u.Utilities.recursiveClone(c||{});return u.Utilities.forEachProperty(r,(t,e)=>{t==="ignoreDuplicateModules"&&typeof a.ignoreDuplicateModules<"u"?a.ignoreDuplicateModules=a.ignoreDuplicateModules.concat(e):t==="paths"&&typeof a.paths<"u"?u.Utilities.forEachProperty(e,(i,s)=>a.paths[i]=s):t==="config"&&typeof a.config<"u"?u.Utilities.forEachProperty(e,(i,s)=>a.config[i]=s):a[t]=u.Utilities.recursiveClone(e)}),m.validateConfigurationOptions(a)}}u.ConfigurationOptionsUtil=m;class f{constructor(r,c){if(this._env=r,this.options=m.mergeConfigurationOptions(c),this._createIgnoreDuplicateModulesMap(),this._createSortedPathsRules(),this.options.baseUrl===""&&this.options.nodeRequire&&this.options.nodeRequire.main&&this.options.nodeRequire.main.filename&&this._env.isNode){let a=this.options.nodeRequire.main.filename,t=Math.max(a.lastIndexOf("/"),a.lastIndexOf("\\"));this.options.baseUrl=a.substring(0,t+1)}}_createIgnoreDuplicateModulesMap(){this.ignoreDuplicateModulesMap={};for(let r=0;r<this.options.ignoreDuplicateModules.length;r++)this.ignoreDuplicateModulesMap[this.options.ignoreDuplicateModules[r]]=!0}_createSortedPathsRules(){this.sortedPathsRules=[],u.Utilities.forEachProperty(this.options.paths,(r,c)=>{Array.isArray(c)?this.sortedPathsRules.push({from:r,to:c}):this.sortedPathsRules.push({from:r,to:[c]})}),this.sortedPathsRules.sort((r,c)=>c.from.length-r.from.length)}cloneAndMerge(r){return new f(this._env,m.mergeConfigurationOptions(r,this.options))}getOptionsLiteral(){return this.options}_applyPaths(r){let c;for(let a=0,t=this.sortedPathsRules.length;a<t;a++)if(c=this.sortedPathsRules[a],u.Utilities.startsWith(r,c.from)){let e=[];for(let i=0,s=c.to.length;i<s;i++)e.push(c.to[i]+r.substr(c.from.length));return e}return[r]}_addUrlArgsToUrl(r){return u.Utilities.containsQueryString(r)?r+"&"+this.options.urlArgs:r+"?"+this.options.urlArgs}_addUrlArgsIfNecessaryToUrl(r){return this.options.urlArgs?this._addUrlArgsToUrl(r):r}_addUrlArgsIfNecessaryToUrls(r){if(this.options.urlArgs)for(let c=0,a=r.length;c<a;c++)r[c]=this._addUrlArgsToUrl(r[c]);return r}moduleIdToPaths(r){if(this._env.isNode&&this.options.amdModulesPattern instanceof RegExp&&!this.options.amdModulesPattern.test(r))return this.isBuild()?["empty:"]:["node|"+r];let c=r,a;if(!u.Utilities.endsWith(c,".js")&&!u.Utilities.isAbsolutePath(c)){a=this._applyPaths(c);for(let t=0,e=a.length;t<e;t++)this.isBuild()&&a[t]==="empty:"||(u.Utilities.isAbsolutePath(a[t])||(a[t]=this.options.baseUrl+a[t]),!u.Utilities.endsWith(a[t],".js")&&!u.Utilities.containsQueryString(a[t])&&(a[t]=a[t]+".js"))}else!u.Utilities.endsWith(c,".js")&&!u.Utilities.containsQueryString(c)&&(c=c+".js"),a=[c];return this._addUrlArgsIfNecessaryToUrls(a)}requireToUrl(r){let c=r;return u.Utilities.isAbsolutePath(c)||(c=this._applyPaths(c)[0],u.Utilities.isAbsolutePath(c)||(c=this.options.baseUrl+c)),this._addUrlArgsIfNecessaryToUrl(c)}isBuild(){return this.options.isBuild}shouldInvokeFactory(r){return!!(!this.options.isBuild||u.Utilities.isAnonymousModule(r)||this.options.buildForceInvokeFactory&&this.options.buildForceInvokeFactory[r])}isDuplicateMessageIgnoredFor(r){return this.ignoreDuplicateModulesMap.hasOwnProperty(r)}getConfigForModule(r){if(this.options.config)return this.options.config[r]}shouldCatchError(){return this.options.catchError}shouldRecordStats(){return this.options.recordStats}onError(r){this.options.onError(r)}}u.Configuration=f})(AMDLoader||(AMDLoader={}));var AMDLoader;(function(u){class _{constructor(e){this._env=e,this._scriptLoader=null,this._callbackMap={}}load(e,i,s,n){if(!this._scriptLoader)if(this._env.isWebWorker)this._scriptLoader=new h;else if(this._env.isElectronRenderer){const{preferScriptTags:d}=e.getConfig().getOptionsLiteral();d?this._scriptLoader=new m:this._scriptLoader=new r(this._env)}else this._env.isNode?this._scriptLoader=new r(this._env):this._scriptLoader=new m;let o={callback:s,errorback:n};if(this._callbackMap.hasOwnProperty(i)){this._callbackMap[i].push(o);return}this._callbackMap[i]=[o],this._scriptLoader.load(e,i,()=>this.triggerCallback(i),d=>this.triggerErrorback(i,d))}triggerCallback(e){let i=this._callbackMap[e];delete this._callbackMap[e];for(let s=0;s<i.length;s++)i[s].callback()}triggerErrorback(e,i){let s=this._callbackMap[e];delete this._callbackMap[e];for(let n=0;n<s.length;n++)s[n].errorback(i)}}class m{attachListeners(e,i,s){let n=()=>{e.removeEventListener("load",o),e.removeEventListener("error",d)},o=l=>{n(),i()},d=l=>{n(),s(l)};e.addEventListener("load",o),e.addEventListener("error",d)}load(e,i,s,n){if(/^node\|/.test(i)){let o=e.getConfig().getOptionsLiteral(),d=c(e.getRecorder(),o.nodeRequire||u.global.nodeRequire),l=i.split("|"),g=null;try{g=d(l[1])}catch(p){n(p);return}e.enqueueDefineAnonymousModule([],()=>g),s()}else{let o=document.createElement("script");o.setAttribute("async","async"),o.setAttribute("type","text/javascript"),this.attachListeners(o,s,n);const{trustedTypesPolicy:d}=e.getConfig().getOptionsLiteral();d&&(i=d.createScriptURL(i)),o.setAttribute("src",i);const{cspNonce:l}=e.getConfig().getOptionsLiteral();l&&o.setAttribute("nonce",l),document.getElementsByTagName("head")[0].appendChild(o)}}}function f(t){const{trustedTypesPolicy:e}=t.getConfig().getOptionsLiteral();try{return(e?self.eval(e.createScript("","true")):new Function("true")).call(self),!0}catch{return!1}}class h{constructor(){this._cachedCanUseEval=null}_canUseEval(e){return this._cachedCanUseEval===null&&(this._cachedCanUseEval=f(e)),this._cachedCanUseEval}load(e,i,s,n){if(/^node\|/.test(i)){const o=e.getConfig().getOptionsLiteral(),d=c(e.getRecorder(),o.nodeRequire||u.global.nodeRequire),l=i.split("|");let g=null;try{g=d(l[1])}catch(p){n(p);return}e.enqueueDefineAnonymousModule([],function(){return g}),s()}else{const{trustedTypesPolicy:o}=e.getConfig().getOptionsLiteral();if(!(/^((http:)|(https:)|(file:))/.test(i)&&i.substring(0,self.origin.length)!==self.origin)&&this._canUseEval(e)){fetch(i).then(l=>{if(l.status!==200)throw new Error(l.statusText);return l.text()}).then(l=>{l=`${l}
//# sourceURL=${i}`,(o?self.eval(o.createScript("",l)):new Function(l)).call(self),s()}).then(void 0,n);return}try{o&&(i=o.createScriptURL(i)),importScripts(i),s()}catch(l){n(l)}}}}class r{constructor(e){this._env=e,this._didInitialize=!1,this._didPatchNodeRequire=!1}_init(e){this._didInitialize||(this._didInitialize=!0,this._fs=e("fs"),this._vm=e("vm"),this._path=e("path"),this._crypto=e("crypto"))}_initNodeRequire(e,i){const{nodeCachedData:s}=i.getConfig().getOptionsLiteral();if(!s||this._didPatchNodeRequire)return;this._didPatchNodeRequire=!0;const n=this,o=e("module");function d(l){const g=l.constructor;let p=function(v){try{return l.require(v)}finally{}};return p.resolve=function(v,E){return g._resolveFilename(v,l,!1,E)},p.resolve.paths=function(v){return g._resolveLookupPaths(v,l)},p.main=process.mainModule,p.extensions=g._extensions,p.cache=g._cache,p}o.prototype._compile=function(l,g){const p=o.wrap(l.replace(/^#!.*/,"")),y=i.getRecorder(),v=n._getCachedDataPath(s,g),E={filename:g};let I;try{const D=n._fs.readFileSync(v);I=D.slice(0,16),E.cachedData=D.slice(16),y.record(60,v)}catch{y.record(61,v)}const R=new n._vm.Script(p,E),b=R.runInThisContext(E),w=n._path.dirname(g),C=d(this),U=[this.exports,C,this,g,w,process,_commonjsGlobal,Buffer],P=b.apply(this.exports,U);return n._handleCachedData(R,p,v,!E.cachedData,i),n._verifyCachedData(R,p,v,I,i),P}}load(e,i,s,n){const o=e.getConfig().getOptionsLiteral(),d=c(e.getRecorder(),o.nodeRequire||u.global.nodeRequire),l=o.nodeInstrumenter||function(p){return p};this._init(d),this._initNodeRequire(d,e);let g=e.getRecorder();if(/^node\|/.test(i)){let p=i.split("|"),y=null;try{y=d(p[1])}catch(v){n(v);return}e.enqueueDefineAnonymousModule([],()=>y),s()}else{i=u.Utilities.fileUriToFilePath(this._env.isWindows,i);const p=this._path.normalize(i),y=this._getElectronRendererScriptPathOrUri(p),v=!!o.nodeCachedData,E=v?this._getCachedDataPath(o.nodeCachedData,i):void 0;this._readSourceAndCachedData(p,E,g,(I,R,b,w)=>{if(I){n(I);return}let C;R.charCodeAt(0)===r._BOM?C=r._PREFIX+R.substring(1)+r._SUFFIX:C=r._PREFIX+R+r._SUFFIX,C=l(C,p);const U={filename:y,cachedData:b},P=this._createAndEvalScript(e,C,U,s,n);this._handleCachedData(P,C,E,v&&!b,e),this._verifyCachedData(P,C,E,w,e)})}}_createAndEvalScript(e,i,s,n,o){const d=e.getRecorder();d.record(31,s.filename);const l=new this._vm.Script(i,s),g=l.runInThisContext(s),p=e.getGlobalAMDDefineFunc();let y=!1;const v=function(){return y=!0,p.apply(null,arguments)};return v.amd=p.amd,g.call(u.global,e.getGlobalAMDRequireFunc(),v,s.filename,this._path.dirname(s.filename)),d.record(32,s.filename),y?n():o(new Error(`Didn't receive define call in ${s.filename}!`)),l}_getElectronRendererScriptPathOrUri(e){if(!this._env.isElectronRenderer)return e;let i=e.match(/^([a-z])\:(.*)/i);return i?`file:///${(i[1].toUpperCase()+":"+i[2]).replace(/\\/g,"/")}`:`file://${e}`}_getCachedDataPath(e,i){const s=this._crypto.createHash("md5").update(i,"utf8").update(e.seed,"utf8").update(process.arch,"").digest("hex"),n=this._path.basename(i).replace(/\.js$/,"");return this._path.join(e.path,`${n}-${s}.code`)}_handleCachedData(e,i,s,n,o){e.cachedDataRejected?this._fs.unlink(s,d=>{o.getRecorder().record(62,s),this._createAndWriteCachedData(e,i,s,o),d&&o.getConfig().onError(d)}):n&&this._createAndWriteCachedData(e,i,s,o)}_createAndWriteCachedData(e,i,s,n){let o=Math.ceil(n.getConfig().getOptionsLiteral().nodeCachedData.writeDelay*(1+Math.random())),d=-1,l=0,g;const p=()=>{setTimeout(()=>{g||(g=this._crypto.createHash("md5").update(i,"utf8").digest());const y=e.createCachedData();if(!(y.length===0||y.length===d||l>=5)){if(y.length<d){p();return}d=y.length,this._fs.writeFile(s,Buffer.concat([g,y]),v=>{v&&n.getConfig().onError(v),n.getRecorder().record(63,s),p()})}},o*Math.pow(4,l++))};p()}_readSourceAndCachedData(e,i,s,n){if(!i)this._fs.readFile(e,{encoding:"utf8"},n);else{let o,d,l,g=2;const p=y=>{y?n(y):--g===0&&n(void 0,o,d,l)};this._fs.readFile(e,{encoding:"utf8"},(y,v)=>{o=v,p(y)}),this._fs.readFile(i,(y,v)=>{!y&&v&&v.length>0?(l=v.slice(0,16),d=v.slice(16),s.record(60,i)):s.record(61,i),p()})}}_verifyCachedData(e,i,s,n,o){n&&(e.cachedDataRejected||setTimeout(()=>{const d=this._crypto.createHash("md5").update(i,"utf8").digest();n.equals(d)||(o.getConfig().onError(new Error(`FAILED TO VERIFY CACHED DATA, deleting stale '${s}' now, but a RESTART IS REQUIRED`)),this._fs.unlink(s,l=>{l&&o.getConfig().onError(l)}))},Math.ceil(5e3*(1+Math.random()))))}}r._BOM=65279,r._PREFIX="(function (require, define, __filename, __dirname) { ",r._SUFFIX=`
});`;function c(t,e){if(e.__$__isRecorded)return e;const i=function(n){t.record(33,n);try{return e(n)}finally{t.record(34,n)}};return i.__$__isRecorded=!0,i}u.ensureRecordedNodeRequire=c;function a(t){return new _(t)}u.createScriptLoader=a})(AMDLoader||(AMDLoader={}));var AMDLoader;(function(u){class _{constructor(t){let e=t.lastIndexOf("/");e!==-1?this.fromModulePath=t.substr(0,e+1):this.fromModulePath=""}static _normalizeModuleId(t){let e=t,i;for(i=/\/\.\//;i.test(e);)e=e.replace(i,"/");for(e=e.replace(/^\.\//g,""),i=/\/(([^\/])|([^\/][^\/\.])|([^\/\.][^\/])|([^\/][^\/][^\/]+))\/\.\.\//;i.test(e);)e=e.replace(i,"/");return e=e.replace(/^(([^\/])|([^\/][^\/\.])|([^\/\.][^\/])|([^\/][^\/][^\/]+))\/\.\.\//,""),e}resolveModule(t){let e=t;return u.Utilities.isAbsolutePath(e)||(u.Utilities.startsWith(e,"./")||u.Utilities.startsWith(e,"../"))&&(e=_._normalizeModuleId(this.fromModulePath+e)),e}}_.ROOT=new _(""),u.ModuleIdResolver=_;class m{constructor(t,e,i,s,n,o){this.id=t,this.strId=e,this.dependencies=i,this._callback=s,this._errorback=n,this.moduleIdResolver=o,this.exports={},this.error=null,this.exportsPassedIn=!1,this.unresolvedDependenciesCount=this.dependencies.length,this._isComplete=!1}static _safeInvokeFunction(t,e){try{return{returnedValue:t.apply(u.global,e),producedError:null}}catch(i){return{returnedValue:null,producedError:i}}}static _invokeFactory(t,e,i,s){return t.shouldInvokeFactory(e)?t.shouldCatchError()?this._safeInvokeFunction(i,s):{returnedValue:i.apply(u.global,s),producedError:null}:{returnedValue:null,producedError:null}}complete(t,e,i,s){this._isComplete=!0;let n=null;if(this._callback)if(typeof this._callback=="function"){t.record(21,this.strId);let o=m._invokeFactory(e,this.strId,this._callback,i);n=o.producedError,t.record(22,this.strId),!n&&typeof o.returnedValue<"u"&&(!this.exportsPassedIn||u.Utilities.isEmpty(this.exports))&&(this.exports=o.returnedValue)}else this.exports=this._callback;if(n){let o=u.ensureError(n);o.phase="factory",o.moduleId=this.strId,o.neededBy=s(this.id),this.error=o,e.onError(o)}this.dependencies=null,this._callback=null,this._errorback=null,this.moduleIdResolver=null}onDependencyError(t){return this._isComplete=!0,this.error=t,this._errorback?(this._errorback(t),!0):!1}isComplete(){return this._isComplete}}u.Module=m;class f{constructor(){this._nextId=0,this._strModuleIdToIntModuleId=new Map,this._intModuleIdToStrModuleId=[],this.getModuleId("exports"),this.getModuleId("module"),this.getModuleId("require")}getMaxModuleId(){return this._nextId}getModuleId(t){let e=this._strModuleIdToIntModuleId.get(t);return typeof e>"u"&&(e=this._nextId++,this._strModuleIdToIntModuleId.set(t,e),this._intModuleIdToStrModuleId[e]=t),e}getStrModuleId(t){return this._intModuleIdToStrModuleId[t]}}class h{constructor(t){this.id=t}}h.EXPORTS=new h(0),h.MODULE=new h(1),h.REQUIRE=new h(2),u.RegularDependency=h;class r{constructor(t,e,i){this.id=t,this.pluginId=e,this.pluginParam=i}}u.PluginDependency=r;class c{constructor(t,e,i,s,n=0){this._env=t,this._scriptLoader=e,this._loaderAvailableTimestamp=n,this._defineFunc=i,this._requireFunc=s,this._moduleIdProvider=new f,this._config=new u.Configuration(this._env),this._hasDependencyCycle=!1,this._modules2=[],this._knownModules2=[],this._inverseDependencies2=[],this._inversePluginDependencies2=new Map,this._currentAnonymousDefineCall=null,this._recorder=null,this._buildInfoPath=[],this._buildInfoDefineStack=[],this._buildInfoDependencies=[],this._requireFunc.moduleManager=this}reset(){return new c(this._env,this._scriptLoader,this._defineFunc,this._requireFunc,this._loaderAvailableTimestamp)}getGlobalAMDDefineFunc(){return this._defineFunc}getGlobalAMDRequireFunc(){return this._requireFunc}static _findRelevantLocationInStack(t,e){let i=o=>o.replace(/\\/g,"/"),s=i(t),n=e.split(/\n/);for(let o=0;o<n.length;o++){let d=n[o].match(/(.*):(\d+):(\d+)\)?$/);if(d){let l=d[1],g=d[2],p=d[3],y=Math.max(l.lastIndexOf(" ")+1,l.lastIndexOf("(")+1);if(l=l.substr(y),l=i(l),l===s){let v={line:parseInt(g,10),col:parseInt(p,10)};return v.line===1&&(v.col-=53),v}}}throw new Error("Could not correlate define call site for needle "+t)}getBuildInfo(){if(!this._config.isBuild())return null;let t=[],e=0;for(let i=0,s=this._modules2.length;i<s;i++){let n=this._modules2[i];if(!n)continue;let o=this._buildInfoPath[n.id]||null,d=this._buildInfoDefineStack[n.id]||null,l=this._buildInfoDependencies[n.id];t[e++]={id:n.strId,path:o,defineLocation:o&&d?c._findRelevantLocationInStack(o,d):null,dependencies:l,shim:null,exports:n.exports}}return t}getRecorder(){return this._recorder||(this._config.shouldRecordStats()?this._recorder=new u.LoaderEventRecorder(this._loaderAvailableTimestamp):this._recorder=u.NullLoaderEventRecorder.INSTANCE),this._recorder}getLoaderEvents(){return this.getRecorder().getEvents()}enqueueDefineAnonymousModule(t,e){if(this._currentAnonymousDefineCall!==null)throw new Error("Can only have one anonymous define call per script file");let i=null;this._config.isBuild()&&(i=new Error("StackLocation").stack||null),this._currentAnonymousDefineCall={stack:i,dependencies:t,callback:e}}defineModule(t,e,i,s,n,o=new _(t)){let d=this._moduleIdProvider.getModuleId(t);if(this._modules2[d]){this._config.isDuplicateMessageIgnoredFor(t)||console.warn("Duplicate definition of module '"+t+"'");return}let l=new m(d,t,this._normalizeDependencies(e,o),i,s,o);this._modules2[d]=l,this._config.isBuild()&&(this._buildInfoDefineStack[d]=n,this._buildInfoDependencies[d]=(l.dependencies||[]).map(g=>this._moduleIdProvider.getStrModuleId(g.id))),this._resolve(l)}_normalizeDependency(t,e){if(t==="exports")return h.EXPORTS;if(t==="module")return h.MODULE;if(t==="require")return h.REQUIRE;let i=t.indexOf("!");if(i>=0){let s=e.resolveModule(t.substr(0,i)),n=e.resolveModule(t.substr(i+1)),o=this._moduleIdProvider.getModuleId(s+"!"+n),d=this._moduleIdProvider.getModuleId(s);return new r(o,d,n)}return new h(this._moduleIdProvider.getModuleId(e.resolveModule(t)))}_normalizeDependencies(t,e){let i=[],s=0;for(let n=0,o=t.length;n<o;n++)i[s++]=this._normalizeDependency(t[n],e);return i}_relativeRequire(t,e,i,s){if(typeof e=="string")return this.synchronousRequire(e,t);this.defineModule(u.Utilities.generateAnonymousModule(),e,i,s,null,t)}synchronousRequire(t,e=new _(t)){let i=this._normalizeDependency(t,e),s=this._modules2[i.id];if(!s)throw new Error("Check dependency list! Synchronous require cannot resolve module '"+t+"'. This is the first mention of this module!");if(!s.isComplete())throw new Error("Check dependency list! Synchronous require cannot resolve module '"+t+"'. This module has not been resolved completely yet.");if(s.error)throw s.error;return s.exports}configure(t,e){let i=this._config.shouldRecordStats();e?this._config=new u.Configuration(this._env,t):this._config=this._config.cloneAndMerge(t),this._config.shouldRecordStats()&&!i&&(this._recorder=null)}getConfig(){return this._config}_onLoad(t){if(this._currentAnonymousDefineCall!==null){let e=this._currentAnonymousDefineCall;this._currentAnonymousDefineCall=null,this.defineModule(this._moduleIdProvider.getStrModuleId(t),e.dependencies,e.callback,null,e.stack)}}_createLoadError(t,e){let i=this._moduleIdProvider.getStrModuleId(t),s=(this._inverseDependencies2[t]||[]).map(o=>this._moduleIdProvider.getStrModuleId(o));const n=u.ensureError(e);return n.phase="loading",n.moduleId=i,n.neededBy=s,n}_onLoadError(t,e){const i=this._createLoadError(t,e);this._modules2[t]||(this._modules2[t]=new m(t,this._moduleIdProvider.getStrModuleId(t),[],()=>{},null,null));let s=[];for(let d=0,l=this._moduleIdProvider.getMaxModuleId();d<l;d++)s[d]=!1;let n=!1,o=[];for(o.push(t),s[t]=!0;o.length>0;){let d=o.shift(),l=this._modules2[d];l&&(n=l.onDependencyError(i)||n);let g=this._inverseDependencies2[d];if(g)for(let p=0,y=g.length;p<y;p++){let v=g[p];s[v]||(o.push(v),s[v]=!0)}}n||this._config.onError(i)}_hasDependencyPath(t,e){let i=this._modules2[t];if(!i)return!1;let s=[];for(let o=0,d=this._moduleIdProvider.getMaxModuleId();o<d;o++)s[o]=!1;let n=[];for(n.push(i),s[t]=!0;n.length>0;){let d=n.shift().dependencies;if(d)for(let l=0,g=d.length;l<g;l++){let p=d[l];if(p.id===e)return!0;let y=this._modules2[p.id];y&&!s[p.id]&&(s[p.id]=!0,n.push(y))}}return!1}_findCyclePath(t,e,i){if(t===e||i===100)return[t];let s=this._modules2[t];if(!s)return null;let n=s.dependencies;if(n)for(let o=0,d=n.length;o<d;o++){let l=this._findCyclePath(n[o].id,e,i+1);if(l!==null)return l.push(t),l}return null}_createRequire(t){let e=(i,s,n)=>this._relativeRequire(t,i,s,n);return e.toUrl=i=>this._config.requireToUrl(t.resolveModule(i)),e.getStats=()=>this.getLoaderEvents(),e.hasDependencyCycle=()=>this._hasDependencyCycle,e.config=(i,s=!1)=>{this.configure(i,s)},e.__$__nodeRequire=u.global.nodeRequire,e}_loadModule(t){if(this._modules2[t]||this._knownModules2[t])return;this._knownModules2[t]=!0;let e=this._moduleIdProvider.getStrModuleId(t),i=this._config.moduleIdToPaths(e),s=/^@[^\/]+\/[^\/]+$/;this._env.isNode&&(e.indexOf("/")===-1||s.test(e))&&i.push("node|"+e);let n=-1,o=d=>{if(n++,n>=i.length)this._onLoadError(t,d);else{let l=i[n],g=this.getRecorder();if(this._config.isBuild()&&l==="empty:"){this._buildInfoPath[t]=l,this.defineModule(this._moduleIdProvider.getStrModuleId(t),[],null,null,null),this._onLoad(t);return}g.record(10,l),this._scriptLoader.load(this,l,()=>{this._config.isBuild()&&(this._buildInfoPath[t]=l),g.record(11,l),this._onLoad(t)},p=>{g.record(12,l),o(p)})}};o(null)}_loadPluginDependency(t,e){if(this._modules2[e.id]||this._knownModules2[e.id])return;this._knownModules2[e.id]=!0;let i=s=>{this.defineModule(this._moduleIdProvider.getStrModuleId(e.id),[],s,null,null)};i.error=s=>{this._config.onError(this._createLoadError(e.id,s))},t.load(e.pluginParam,this._createRequire(_.ROOT),i,this._config.getOptionsLiteral())}_resolve(t){let e=t.dependencies;if(e)for(let i=0,s=e.length;i<s;i++){let n=e[i];if(n===h.EXPORTS){t.exportsPassedIn=!0,t.unresolvedDependenciesCount--;continue}if(n===h.MODULE){t.unresolvedDependenciesCount--;continue}if(n===h.REQUIRE){t.unresolvedDependenciesCount--;continue}let o=this._modules2[n.id];if(o&&o.isComplete()){if(o.error){t.onDependencyError(o.error);return}t.unresolvedDependenciesCount--;continue}if(this._hasDependencyPath(n.id,t.id)){this._hasDependencyCycle=!0,console.warn("There is a dependency cycle between '"+this._moduleIdProvider.getStrModuleId(n.id)+"' and '"+this._moduleIdProvider.getStrModuleId(t.id)+"'. The cyclic path follows:");let d=this._findCyclePath(n.id,t.id,0)||[];d.reverse(),d.push(n.id),console.warn(d.map(l=>this._moduleIdProvider.getStrModuleId(l)).join(` => 
`)),t.unresolvedDependenciesCount--;continue}if(this._inverseDependencies2[n.id]=this._inverseDependencies2[n.id]||[],this._inverseDependencies2[n.id].push(t.id),n instanceof r){let d=this._modules2[n.pluginId];if(d&&d.isComplete()){this._loadPluginDependency(d.exports,n);continue}let l=this._inversePluginDependencies2.get(n.pluginId);l||(l=[],this._inversePluginDependencies2.set(n.pluginId,l)),l.push(n),this._loadModule(n.pluginId);continue}this._loadModule(n.id)}t.unresolvedDependenciesCount===0&&this._onModuleComplete(t)}_onModuleComplete(t){let e=this.getRecorder();if(t.isComplete())return;let i=t.dependencies,s=[];if(i)for(let l=0,g=i.length;l<g;l++){let p=i[l];if(p===h.EXPORTS){s[l]=t.exports;continue}if(p===h.MODULE){s[l]={id:t.strId,config:()=>this._config.getConfigForModule(t.strId)};continue}if(p===h.REQUIRE){s[l]=this._createRequire(t.moduleIdResolver);continue}let y=this._modules2[p.id];if(y){s[l]=y.exports;continue}s[l]=null}const n=l=>(this._inverseDependencies2[l]||[]).map(g=>this._moduleIdProvider.getStrModuleId(g));t.complete(e,this._config,s,n);let o=this._inverseDependencies2[t.id];if(this._inverseDependencies2[t.id]=null,o)for(let l=0,g=o.length;l<g;l++){let p=o[l],y=this._modules2[p];y.unresolvedDependenciesCount--,y.unresolvedDependenciesCount===0&&this._onModuleComplete(y)}let d=this._inversePluginDependencies2.get(t.id);if(d){this._inversePluginDependencies2.delete(t.id);for(let l=0,g=d.length;l<g;l++)this._loadPluginDependency(t.exports,d[l])}}}u.ModuleManager=c})(AMDLoader||(AMDLoader={}));var define,AMDLoader;(function(u){const _=new u.Environment;let m=null;const f=function(a,t,e){typeof a!="string"&&(e=t,t=a,a=null),(typeof t!="object"||!Array.isArray(t))&&(e=t,t=null),t||(t=["require","exports","module"]),a?m.defineModule(a,t,e,null,null):m.enqueueDefineAnonymousModule(t,e)};f.amd={jQuery:!0};const h=function(a,t=!1){m.configure(a,t)},r=function(){if(arguments.length===1){if(arguments[0]instanceof Object&&!Array.isArray(arguments[0])){h(arguments[0]);return}if(typeof arguments[0]=="string")return m.synchronousRequire(arguments[0])}if((arguments.length===2||arguments.length===3)&&Array.isArray(arguments[0])){m.defineModule(u.Utilities.generateAnonymousModule(),arguments[0],arguments[1],arguments[2],null);return}throw new Error("Unrecognized require call")};r.config=h,r.getConfig=function(){return m.getConfig().getOptionsLiteral()},r.reset=function(){m=m.reset()},r.getBuildInfo=function(){return m.getBuildInfo()},r.getStats=function(){return m.getLoaderEvents()},r.define=f;function c(){if(typeof u.global.require<"u"||typeof require<"u"){const a=u.global.require||require;if(typeof a=="function"&&typeof a.resolve=="function"){const t=u.ensureRecordedNodeRequire(m.getRecorder(),a);u.global.nodeRequire=t,r.nodeRequire=t,r.__$__nodeRequire=t}}_.isNode&&!_.isElectronRenderer&&!_.isElectronNodeIntegrationWebWorker?module.exports=r:(_.isElectronRenderer||(u.global.define=f),u.global.require=r)}u.init=c,(typeof u.global.define!="function"||!u.global.define.amd)&&(m=new u.ModuleManager(_,u.createScriptLoader(_),f,r,u.Utilities.getHighPerformanceTimestamp()),typeof u.global.require<"u"&&typeof u.global.require!="function"&&r.config(u.global.require),define=function(){return f.apply(null,arguments)},define.amd=f.amd,typeof doNotInitLoader>"u"&&c())})(AMDLoader||(AMDLoader={})),define("vs/css",["require","exports"],function(u,_){"use strict";Object.defineProperty(_,"__esModule",{value:!0}),_.load=m;function m(a,t,e,i){if(i=i||{},(i["vs/css"]||{}).disabled){e({});return}const n=t.toUrl(a+".css");f(a,n,()=>{e({})},o=>{typeof e.error=="function"&&e.error("Could not find "+n+".")})}function f(a,t,e,i){if(h(a,t)){e();return}r(a,t,e,i)}function h(a,t){const e=window.document.getElementsByTagName("link");for(let i=0,s=e.length;i<s;i++){const n=e[i].getAttribute("data-name"),o=e[i].getAttribute("href");if(n===a||o===t)return!0}return!1}function r(a,t,e,i){const s=document.createElement("link");s.setAttribute("rel","stylesheet"),s.setAttribute("type","text/css"),s.setAttribute("data-name",a),c(a,s,e,i),s.setAttribute("href",t),(window.document.head||window.document.getElementsByTagName("head")[0]).appendChild(s)}function c(a,t,e,i){const s=()=>{t.removeEventListener("load",n),t.removeEventListener("error",o)},n=d=>{s(),e()},o=d=>{s(),i(d)};t.addEventListener("load",n),t.addEventListener("error",o)}});

//# sourceMappingURL=https://cursor-sourcemaps.s3.amazonaws.com/sourcemaps/2eaa79a1b14ccff5d1c78a2c358a08be16a8e5a0/core/vs/loader.js.map
